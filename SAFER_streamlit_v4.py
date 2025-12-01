import os, re, json, random, sqlite3, logging, uuid, asyncio, math
from datetime import datetime
from typing import Dict, List, Optional, Any
import streamlit as st
import nbformat, nest_asyncio
nest_asyncio.apply()

logger = logging.getLogger(__name__)
logging.getLogger("asyncio").setLevel(logging.ERROR)

RERUN_AVAILABLE = hasattr(st, "experimental_rerun")

# ADK imports optional
try:
    from google.adk.agents import LlmAgent  # type: ignore
    from google.adk.models.google_llm import Gemini  # type: ignore
    from google.adk.runners import Runner  # type: ignore
    from google.adk.sessions import InMemorySessionService  # type: ignore
    from google.genai import types  # type: ignore
    ADK_AVAILABLE = True
except Exception as e:
    logger.info("ADK import failed: %s", e)
    ADK_AVAILABLE = False

# Paths & DB
HOUSEHOLDS_DIR = "households"
DB_NAME = "financial_protocols.db"
NOTEBOOK_PATH = "/mnt/data/notebook32601a58a7.ipynb"
os.makedirs(HOUSEHOLDS_DIR, exist_ok=True)

# -------------------- Household helpers --------------------
def household_file_path(hid: str) -> str:
    return os.path.join(HOUSEHOLDS_DIR, f"{hid}.json")

def save_household(profile: Dict[str, Any]) -> None:
    with open(household_file_path(profile["id"]), "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

def load_household(hid: str) -> Optional[Dict[str, Any]]:
    p = household_file_path(hid)
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def list_households() -> List[str]:
    return sorted([f[:-5] for f in os.listdir(HOUSEHOLDS_DIR) if f.endswith(".json")])

def delete_household(hid: str) -> bool:
    p = household_file_path(hid)
    if os.path.exists(p):
        os.remove(p)
        return True
    return False

# Seed sample households
def ensure_samples():
    if not list_households():
        s1 = {"id":"hh_001","name":"Ana","baseline_balance":1500.0,"monthly_income":8000.0,"monthly_expenses":7000.0,"safety_net_contacts":{"community_worker":"CW-001"},"preferences":{"communication_channel":"sms"},"credit_history":{"score":480}}
        s2 = {"id":"hh_002","name":"Ben","baseline_balance":400.0,"monthly_income":3000.0,"monthly_expenses":2900.0,"safety_net_contacts":{"community_worker":"CW-002"},"preferences":{"communication_channel":"sms"},"credit_history":{"score":620}}
        save_household(s1); save_household(s2)
ensure_samples()

# -------------------- Protocol DB & RAG --------------------
def initialize_financial_protocols_db(db_name: str = DB_NAME) -> None:
    conn = sqlite3.connect(db_name); cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS protocols (id INTEGER PRIMARY KEY, protocol_type TEXT, severity TEXT, trigger_value REAL, protocol_text TEXT)""")
    cur.execute("SELECT COUNT(*) FROM protocols")
    if cur.fetchone()[0] == 0:
        entries = [
            ("liquidity","mild",500.0,"Advise budget tightening and defer non-essential expenses."),
            ("liquidity","urgent",200.0,"Urgent: connect to emergency transfer program."),
            ("liquidity","crisis",50.0,"Crisis: notify community worker immediately."),
            ("fraud","mild",30.0,"Possible suspicious activity. Ask user to verify."),
            ("fraud","urgent",70.0,"High fraud likelihood. Freeze account and notify."),
            ("fraud","crisis",90.0,"Confirmed fraud. Escalate to fraud investigations."),
        ]
        cur.executemany("INSERT INTO protocols (protocol_type,severity,trigger_value,protocol_text) VALUES (?,?,?,?)", entries)
        conn.commit()
    conn.close()

initialize_financial_protocols_db()

def search_financial_protocols(snapshot: str) -> str:
    conn = sqlite3.connect(DB_NAME); cur = conn.cursor(); outputs: List[str]=[]
    fm = re.search(r"FraudScore=(\d+)", snapshot)
    if fm:
        fs = int(fm.group(1))
        cur.execute("SELECT severity, protocol_text FROM protocols WHERE protocol_type='fraud' AND trigger_value <= ? ORDER BY trigger_value DESC", (fs,))
        outputs += [f"[FRAUD - {sev}] {txt}" for sev,txt in cur.fetchall()]
    bm = re.search(r"Balance\s*(\d+\.?\d*)", snapshot)
    if bm:
        bal = float(bm.group(1))
        cur.execute("SELECT severity, trigger_value, protocol_text FROM protocols WHERE protocol_type='liquidity' ORDER BY trigger_value DESC")
        for sev,tv,txt in cur.fetchall():
            if bal <= tv:
                outputs.append(f"[LIQUIDITY - {sev}] {txt}")
    conn.close()
    return "\n".join(outputs) if outputs else "No protocols triggered. Continue monitoring."

# -------------------- Core tools & agents --------------------
def suggest_budget_actions(hh: Dict[str,Any]) -> str:
    tips = []
    if hh["monthly_expenses"] > hh["monthly_income"]:
        tips.append("Expenses exceed income. Prioritize essentials and seek support.")
    else:
        tips.append("Within income. Build a 1-month buffer.")
    if hh["baseline_balance"] < 500:
        tips.append("Balance low: consider community programs.")
    return "Budget Tips:\n" + "\n".join(tips)

def get_monthly_statement(hh: Dict[str,Any], month: str) -> str:
    balance = hh.get("baseline_balance",0.0); txs = []
    for i in range(6):
        amt = round(random.uniform(-1200,1200),2); balance += amt
        txs.append(f"{i+1}. {'CR' if amt>0 else 'DB'} {abs(amt):.2f} -> bal {balance:.2f}")
    return f"Monthly Statement ({month}) for {hh.get('name')}:\n" + "\n".join(txs)

def request_microloan(hh: Dict[str,Any], amount: float) -> str:
    score = hh.get("credit_history",{}).get("score",0)
    if score >= 600 and amount <= 0.5 * hh.get("monthly_income",0):
        return f"✅ Loan approved for {hh.get('name')}: PHP {amount:.2f}"
    return f"❌ Loan declined for {hh.get('name')}. Consider smaller amount or improving credit score."

def simulate_transaction_stream(hh: Dict[str,Any]) -> str:
    if random.random() < 0.7:
        change = random.uniform(-200,200); hh["baseline_balance"] = round(hh.get("baseline_balance",0.0) + change,2)
        return f"Snapshot: Balance {hh['baseline_balance']:.2f}. Normal activity."
    fraud_score = random.randint(0,100); drop=random.uniform(200,2500); hh["baseline_balance"] = round(max(0.0, hh.get("baseline_balance",0.0)-drop),2)
    return f"ANOMALY: Large debit {drop:.2f}. Balance {hh['baseline_balance']:.2f}. FraudScore={fraud_score}"

# ADK tool wrappers
def budgetcoach_tool(hid: str) -> str:
    hh = load_household(hid); return suggest_budget_actions(hh) if hh else "Household not found."

def creditadvisor_tool(hid: str, amount: Optional[float] = None) -> str:
    hh = load_household(hid)
    if not hh: return "Household not found."
    if amount is None:
        return "CreditAdvisor: provide guidance on safe borrowing and alternatives."
    return request_microloan(hh, amount)

def fraudwatcher_tool(hid: str) -> str:
    hh = load_household(hid); return simulate_transaction_stream(hh) + "\n\n" + search_financial_protocols(simulate_transaction_stream(hh)) if hh else "Household not found."

# ---------- UI helpers for BudgetCoach & FraudWatcher ----------
def run_budgetcoach(household_id: str, use_adk: bool = True) -> Dict[str,str]:
    hh = load_household(household_id)
    if not hh:
        return {"source":"error","advice":"Household not found.","rag":""}
    if ADK_AVAILABLE and use_adk:
        try:
            prompt = f"[household_id={household_id}] Please analyze the household's finances and produce short budget coaching points. Route to BudgetCoach."
            adk_out = run_guardian_sync(prompt)
            return {"source":"adk_budgetcoach","advice":adk_out,"rag":""}
        except Exception as e:
            logger.exception("ADK BudgetCoach failed: %s", e)
    advice = suggest_budget_actions(hh)
    statement = get_monthly_statement(hh, datetime.now().strftime("%B %Y"))
    return {"source":"simulated_budgetcoach", "advice": advice + "\n\n" + statement, "rag":""}

def run_fraudwatcher(household_id: str, use_adk: bool = True) -> Dict[str,str]:
    hh = load_household(household_id)
    if not hh:
        return {"source":"error","advice":"Household not found.","rag":""}
    if ADK_AVAILABLE and use_adk:
        try:
            prompt = f"[household_id={household_id}] Check recent transactions for anomalies and recommend actions. Route to FraudWatcher."
            adk_out = run_guardian_sync(prompt)
            return {"source":"adk_fraudwatcher","advice":adk_out,"rag":""}
        except Exception as e:
            logger.exception("ADK FraudWatcher failed: %s", e)
    snap = simulate_transaction_stream(hh)
    rag = search_financial_protocols(snap)
    actions = []
    if "crisis" in rag.lower():
        cw = hh.get("safety_net_contacts",{}).get("community_worker","CW-UNKNOWN")
        actions.append(notify_community_worker(cw, f"CRISIS for {hh.get('name')}: {snap}"))
        actions.append(send_household_alert(household_id, "Immediate help required. Community worker notified."))
    elif "urgent" in rag.lower():
        actions.append(send_household_alert(household_id, "Urgent issue detected. Please verify transactions."))
    elif "fraud" in rag.lower():
        actions.append(send_household_alert(household_id, "Potential fraud detected. Please verify recent transactions."))
    else:
        actions.append("No escalation required — continue normal monitoring.")
    advice_text = f"{snap}\n\nRAG:\n{rag}\n\nActions:\n" + "\n".join(actions)
    return {"source":"simulated_fraudwatcher","advice":advice_text,"rag":rag}

# -------------------- Notification mocks --------------------
def send_household_alert(hid: str, message: str) -> str:
    hh = load_household(hid)
    if not hh: return "Household not found."
    log = f"[HOUSEHOLD ALERT] To {hh.get('name')} ({hid}): {message}"; logger.info(log); return log

def notify_community_worker(contact_id: str, message: str) -> str:
    log = f"[COMMUNITY WORKER NOTIFY] To {contact_id}: {message}"; logger.info(log); return log

# -------------------- ADK Agents (named) --------------------
BudgetCoach = None; CreditAdvisor = None; FraudWatcher = None; GuardianOrchestrator = None; retry_config=None; gemini_model=None

if ADK_AVAILABLE:
    retry_config = types.HttpRetryOptions(attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429,500,503,504])
    gemini_model = Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config)
    BudgetCoach = LlmAgent(name="BudgetCoach", description="Budget specialist", model=gemini_model, tools=[budgetcoach_tool, get_monthly_statement], instruction="Provide budget advice and near-term liquidity forecasts.")
    CreditAdvisor = LlmAgent(name="CreditAdvisor", description="Credit advisor", model=gemini_model, tools=[creditadvisor_tool], instruction="Advise on safe borrowing and repayment implications.")
    FraudWatcher = LlmAgent(name="FraudWatcher", description="Fraud watcher", model=gemini_model, tools=[fraudwatcher_tool, search_financial_protocols, get_monthly_statement], instruction="Detect anomalies and suggest escalation.")
    GuardianOrchestrator = LlmAgent(name="GuardianOrchestrator", description="Orchestrator", model=gemini_model, instruction="Route queries to subagents (BudgetCoach, CreditAdvisor, FraudWatcher).", sub_agents=[BudgetCoach, CreditAdvisor, FraudWatcher])
    async def _run_guardian_async(user_id: str, session_id: str, query_text: str) -> str:
        session_service = InMemorySessionService()
        runner = Runner(agent=GuardianOrchestrator, app_name="guardian_finance_ui", session_service=session_service)
        await session_service.create_session(app_name="guardian_finance_ui", user_id=user_id, session_id=session_id)
        content = types.Content(parts=[types.Part(text=query_text)])
        output = ""
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            if event.is_final_response() and event.content:
                for part in getattr(event.content, "parts", []):
                    if hasattr(part, "text") and part.text:
                        output += part.text + "\n"
        return output.strip()
    def run_guardian_sync(query_text: str, user_id: str = "streamlit_user") -> str:
        session_id = f"st_{uuid.uuid4().hex[:8]}"
        return asyncio.get_event_loop().run_until_complete(_run_guardian_async(user_id=user_id, session_id=session_id, query_text=query_text))

# -------------------- Notebook prompt extraction (best-effort) --------------------
def extract_prompt_templates(nb_path: str) -> Dict[str,str]:
    templates={}
    if not os.path.exists(nb_path): return templates
    try:
        nb = nbformat.read(nb_path, as_version=4)
        for cell in nb.cells:
            if cell.cell_type=="markdown" and ("prompt" in cell.source.lower() or "advice" in cell.source.lower()):
                templates.setdefault("markdown_hint",""); templates["markdown_hint"] += "\n" + cell.source.strip()
            if cell.cell_type=="code" and ("PROMPT" in cell.source or "generate_advice" in cell.source):
                templates.setdefault("code_template",""); templates["code_template"] += "\n" + cell.source
    except Exception as e:
        logger.exception("Failed to parse notebook: %s", e)
    return templates

NB_TEMPLATES = extract_prompt_templates(NOTEBOOK_PATH)

def build_prompt_from_notebook(hid: str, query: str, templates: Dict[str,str]) -> Optional[str]:
    hh = load_household(hid)
    if not hh: return None
    if "code_template" in templates: return f"Use helper code:\\n{templates['code_template']}\\nHousehold:{json.dumps({'id':hh['id'],'balance':hh.get('baseline_balance')})}\\nQuery:{query}"
    if "markdown_hint" in templates: return templates["markdown_hint"] + f"\\n\\nContext: [household_id={hid}] Query: {query}"
    return None

# -------------------- CreditAdvisor helpers & UI logic --------------------
def monthly_payment(principal: float, annual_rate_pct: float, months: int) -> float:
    if months <= 0: return 0.0
    r = annual_rate_pct / 100.0 / 12.0
    if r == 0:
        return principal / months
    denom = 1 - (1 + r) ** (-months)
    return principal * r / denom if denom != 0 else principal / months

def credit_risk_indicator(hh: Dict[str,Any], amount: float, monthly_payment_est: float) -> str:
    income = max(1.0, hh.get("monthly_income", 0.0))
    expenses = hh.get("monthly_expenses", 0.0)
    dti = (monthly_payment_est + expenses) / income
    score = hh.get("credit_history", {}).get("score", 0)
    if score < 500 or dti > 0.6:
        return "High risk: avoid borrowing if alternatives exist."
    if score < 600 or dti > 0.4:
        return "Medium risk: small, short-term safe loans only."
    return "Low risk: borrowing likely manageable with on-time repayments."

def list_safe_alternatives(hh: Dict[str,Any], amount: float) -> List[str]:
    alts = []
    if hh.get("credit_history", {}).get("score", 0) >= 600:
        alts.append("Formal microfinance institution (lower rates, documented terms)")
    alts.append("Community savings & loan groups (peer-backed loans)")
    if hh.get("baseline_balance", 0) > amount * 0.2:
        alts.append("Use partial savings + small loan to reduce interest costs")
    alts.append("Seek small emergency grant from local social services (if eligible)")
    alts.append("Avoid informal lenders with unclear rates — high risk")
    return alts

# -------------------- High-level generate_advice (uses CreditAdvisor) --------------------
def generate_advice(hid: str, query: str, prefer_notebook: bool = True, use_adk: bool = True) -> Dict[str,Any]:
    hh = load_household(hid)
    if not hh: return {"source":"error","advice":"Household not found.","rag":""}
    try:
        if prefer_notebook and NB_TEMPLATES:
            prompt = build_prompt_from_notebook(hid, query, NB_TEMPLATES)
            if prompt:
                if ADK_AVAILABLE and use_adk:
                    try:
                        out = run_guardian_sync(prompt)
                        return {"source":"notebook+adk","advice":out,"rag":""}
                    except Exception as e:
                        logger.exception("ADK failed: %s", e)
                        return {"source":"notebook+draft","advice":prompt,"rag":""}
                else:
                    return {"source":"notebook+draft","advice":prompt,"rag":""}
    except Exception as e:
        logger.exception("Notebook flow failed: %s", e)
    if ADK_AVAILABLE and use_adk:
        try:
            out = run_guardian_sync(f"[household_id={hid}] {query}")
            return {"source":"adk","advice":out,"rag":""}
        except Exception as e:
            logger.exception("ADK guardian failed: %s", e)
    q = query.lower()
    if "loan" in q or "borrow" in q or "microloan" in q:
        m = re.search(r"\d+\.?\d*", q); amount = float(m.group(0)) if m else 1000.0
        months = 6; rate = 18.0; mp = monthly_payment(amount, rate, months)
        advice = f"CreditAdvisor (simulated): Estimated monthly payment for PHP {amount:.2f} over {months} months at {rate:.1f}% APR is PHP {mp:.2f}.\n"
        advice += credit_risk_indicator(hh, amount, mp) + "\n"
        alts = list_safe_alternatives(hh, amount)
        advice += "\nAlternatives:\n" + "\n".join(f"- {a}" for a in alts)
        return {"source":"simulated_creditadvisor","advice":advice,"rag":""}
    if any(w in q for w in ["balance","budget","statement","monthly","money","expenses"]):
        return {"source":"simulated","advice":suggest_budget_actions(hh) + "\n\n" + get_monthly_statement(hh, datetime.now().strftime("%B %Y")),"rag":""}
    if any(w in q for w in ["suspicious","fraud","debit","anomaly"]):
        snap = simulate_transaction_stream(hh); rag = search_financial_protocols(snap); return {"source":"simulated","advice":snap + "\n\n" + rag,"rag":rag}
    return {"source":"simulated","advice":"I didn't understand. Try: 'balance', 'loan 1000', or 'suspicious'.","rag":""}

# -------------------- ProactiveMonitor (manual multi-cycle) --------------------
def run_proactive_monitor(hids: List[str], cycles: int = 3, shock_chance: float = 0.2) -> List[str]:
    logs: List[str] = []
    logs.append(f"ProactiveMonitor start: {len(hids)} households, cycles={cycles}, shock_chance={shock_chance}")
    for c in range(1, cycles+1):
        logs.append(f"--- Cycle {c} ---")
        for hid in hids:
            hh = load_household(hid)
            if not hh:
                logs.append(f"{hid}: no profile")
                continue
            if random.random() < shock_chance:
                shock_pct = random.uniform(0.1, 0.5)
                drop = round(hh.get("baseline_balance",0.0)*shock_pct,2)
                hh["baseline_balance"] = round(max(0.0, hh.get("baseline_balance",0.0)-drop),2)
                logs.append(f"{hid}: applied shock -PHP {drop:.2f} (new bal {hh['baseline_balance']:.2f})")
            snap = simulate_transaction_stream(hh); rag = search_financial_protocols(snap)
            logs.append(f"{hid}: {snap} | RAG -> {rag}")
            if "crisis" in rag.lower():
                cw = hh.get("safety_net_contacts",{}).get("community_worker","CW-UNKNOWN")
                logs.append(notify_community_worker(cw, f"CRISIS for {hh.get('name')}: {snap}")); logs.append(send_household_alert(hid, "Immediate help required."))
            elif "urgent" in rag.lower():
                logs.append(send_household_alert(hid, "Urgent issue detected. Please verify transactions."))
            elif "fraud" in rag.lower():
                logs.append(send_household_alert(hid, "Potential fraud detected. Please verify recent transactions."))
            else:
                logs.append(f"{hid}: no action required")
    logs.append("ProactiveMonitor complete.")
    return logs

# -------------------- Streamlit UI: full panels --------------------
st.set_page_config(page_title="SAFER", layout="wide")
st.title("SAFER")
st.subheader("Support Agents for Financial Early-warning and Resilience")


# Sidebar policy knobs (regulatory testing)
with st.sidebar:
    st.header("Policy simulation & settings")
    interest_cap = st.slider("Regulatory interest cap (APR %)", min_value=0.0, max_value=100.0, value=36.0, step=1.0)
    min_credit_score_safe = st.slider("Min credit score for 'safe' loans", min_value=300, max_value=850, value=600)
    st.caption("These knobs influence the CreditAdvisor outputs and warnings. They are for regulator-style experiments.")

col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Households")
    ids = list_households()
    if not ids:
        st.info("No households. Create one")
    selected = st.selectbox("Choose household", options=ids)
    if selected:
        hh = load_household(selected)
        if hh:
            st.metric("Balance", f"PHP {hh.get('baseline_balance',0):,.2f}")
            st.write(f"Name: **{hh.get('name')}**"); st.write(f"Income: PHP {hh.get('monthly_income',0):,.2f}")
            st.write(f"Expenses: PHP {hh.get('monthly_expenses',0):,.2f}"); st.write(f"Credit score: {hh.get('credit_history',{}).get('score','N/A')}")

    st.markdown("---")
    st.subheader("Create household")
    with st.form("create", clear_on_submit=True):
        nid = st.text_input("ID (e.g., hh_003)"); nname=st.text_input("Name"); nbal=st.number_input("Baseline balance", value=1000.0); nic=st.number_input("Monthly income", value=5000.0); nexp=st.number_input("Monthly expenses", value=4000.0); nscore=st.number_input("Credit score", value=500)
        create = st.form_submit_button("Create")
        if create:
            if not nid: st.error("Provide ID")
            else:
                save_household({"id":nid,"name":nname or nid,"baseline_balance":float(nbal),"monthly_income":float(nic),"monthly_expenses":float(nexp),"safety_net_contacts":{"community_worker":f"CW-{nid}"},"preferences":{"communication_channel":"sms"},"credit_history":{"score":int(nscore)}})
                st.success("Created"); 
                if RERUN_AVAILABLE:
                    st.experimental_rerun()

with col2:
    st.subheader("Ask the Guardian (conversational)")
    st.write("You can ask about budgets, loans, or suspicious transactions. Quick credit prompts available below.")

    # Read prefill if present (compatibility)
    q_prefill = st.session_state.pop("q_prefill", None) if "q_prefill" in st.session_state else None

    # Input box for conversational queries; if prefilled, populate it
    if q_prefill:
        q = st.text_input("Your question", value=q_prefill, key="main_query")
    else:
        q = st.text_input("Your question", value="Check my balance and suggest budget tips", key="main_query")

    # Quick credit buttons integrated
    col_q1, col_q2, col_q3 = st.columns([1,1,1])
    with col_q1:
        if st.button("Can I afford a loan?"):
            st.session_state["q_prefill"] = "Can I afford a loan of 200000?"
            if RERUN_AVAILABLE: st.experimental_rerun()
    with col_q2:
        if st.button("Explain repayment for 5000"):
            st.session_state["q_prefill"] = "Explain repayment for 5000 over 6 months"
            if RERUN_AVAILABLE: st.experimental_rerun()
    with col_q3:
        if st.button("Should I borrow for emergency?"):
            st.session_state["q_prefill"] = "Should I borrow 3000 for an emergency?"
            if RERUN_AVAILABLE: st.experimental_rerun()

    prefer_nb = st.checkbox("Prefer notebook prompts (advanced)", value=True)
    use_adk = st.checkbox("Use ADK agents (if available)", value=True)

    if st.button("Ask"):
        if not selected:
            st.warning("Please select a household first on the left.")
        else:
            q_val = st.session_state.get("main_query", "Check my balance and suggest budget tips")
            res = generate_advice(selected, q_val, prefer_notebook=prefer_nb, use_adk=use_adk)
            st.session_state.setdefault("history", []).insert(0, {"time":datetime.now().isoformat(timespec="seconds"), "household":selected, "query":q_val, "source":res.get("source"), "advice":res.get("advice"), "rag":res.get("rag","")})
            st.markdown("**Response**"); st.write(f"Source: `{res.get('source')}`"); st.code(res.get("advice") or "No advice generated")
            if res.get("rag"):
                with st.expander("RAG / Protocols"): st.text(res.get("rag"))

    # ---------- Specialist panels (BudgetCoach & FraudWatcher) ----------
    st.markdown("---")
    st.subheader("Specialist Tools: BudgetCoach & FraudWatcher")


    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.markdown("**BudgetCoach**")
        st.write("Get quick budget tips and a short monthly statement.")
        bc_use_adk = st.checkbox("Use ADK for BudgetCoach (if available)", value=True, key="bc_use_adk")
        if st.button("Run BudgetCoach"):
            if not selected:
                st.warning("Select a household first")
            else:
                bc_out = run_budgetcoach(selected, use_adk=bc_use_adk)
                st.write(f"Source: `{bc_out.get('source')}`")
                st.code(bc_out.get("advice", "No output"))
                st.session_state.setdefault("history", []).insert(0, {"time":datetime.now().isoformat(timespec="seconds"), "household":selected, "query":"BudgetCoach", "source":bc_out.get("source"), "advice":bc_out.get("advice"), "rag":bc_out.get("rag","")})

    with col_b2:
        st.markdown("**FraudWatcher**")
        st.write("Check for anomalies and run the RAG protocols.")
        fw_use_adk = st.checkbox("Use ADK for FraudWatcher (if available)", value=True, key="fw_use_adk")
        if st.button("Run FraudWatcher"):
            if not selected:
                st.warning("Select a household first")
            else:
                fw_out = run_fraudwatcher(selected, use_adk=fw_use_adk)
                st.write(f"Source: `{fw_out.get('source')}`")
                if fw_out.get("rag"):
                    with st.expander("RAG / Protocols"):
                        st.text(fw_out.get("rag"))
                st.code(fw_out.get("advice", "No output"))
                st.session_state.setdefault("history", []).insert(0, {"time":datetime.now().isoformat(timespec="seconds"), "household":selected, "query":"FraudWatcher", "source":fw_out.get("source"), "advice":fw_out.get("advice"), "rag":fw_out.get("rag","")})

    st.markdown("---")
    st.subheader("CreditAdvisor")
    st.write("Use this panel for in-depth loan evaluation, repayment simulations, alternatives, and stress tests.")

    # Credit advisor inputs
    loan_amount = st.number_input("Loan amount (PHP)", min_value=100.0, value=2000.0, step=100.0, format="%.2f", key="loan_amount")
    term_months = st.number_input("Term (months)", min_value=1, max_value=60, value=6, key="term_months")
    apr = st.number_input("Interest rate (APR %)", min_value=0.0, max_value=100.0, value=24.0, step=0.5, key="apr")
    stress_income_drop = st.slider("Stress test: income drop %", min_value=0, max_value=100, value=30, key="stress_drop")
    if st.button("Evaluate loan"):
        hh = load_household(selected) if selected else None
        if not hh:
            st.warning("Select a household first")
        else:
            monthly = monthly_payment(loan_amount, apr, term_months)
            risk = credit_risk_indicator(hh, loan_amount, monthly)
            approved = (hh.get("credit_history",{}).get("score",0) >= min_credit_score_safe) and (apr <= interest_cap)
            st.markdown("**Loan evaluation**")
            st.write(f"Monthly payment: PHP {monthly:,.2f} for {term_months} months at {apr:.2f}% APR")
            st.write(f"Estimated risk: {risk}")
            st.write("Policy checks:")
            st.write(f"- Interest cap ({interest_cap:.1f}%): {'PASS' if apr <= interest_cap else 'FAIL'}")
            st.write(f"- Min safe credit score ({min_credit_score_safe}): {'PASS' if hh.get('credit_history',{}).get('score',0) >= min_credit_score_safe else 'FAIL'}")
            st.write(f"→ Overall approval suggestion: {'Likely acceptable' if approved else 'Not recommended under current policy settings'}")
            st.markdown("**Repayment simulation (stress scenario)**")
            income = hh.get("monthly_income",0.0)
            income_after_stress = income * (1 - stress_income_drop/100.0)
            dti = (monthly + hh.get("monthly_expenses",0.0)) / income if income>0 else float('inf')
            dti_stress = (monthly + hh.get("monthly_expenses",0.0)) / max(1.0, income_after_stress)
            st.write(f"DTI normal: {dti:.2f} (monthly payment + expenses) / income")
            st.write(f"DTI under stress ({stress_income_drop}% income drop): {dti_stress:.2f}")
            st.markdown("**Safe alternatives**")
            alts = list_safe_alternatives(hh, loan_amount)
            for a in alts: st.write(f"- {a}")
            if apr > interest_cap:
                st.warning("This loan's APR exceeds the configured regulatory interest cap. Consider safer alternatives.")
            if hh.get("credit_history",{}).get("score",0) < min_credit_score_safe:
                st.warning("Household credit score below the 'safe' threshold. Recommend alternatives or smaller amounts.")

    st.markdown("---")
    st.subheader("History & export")
    hist = st.session_state.get("history", [])
    if not hist:
        st.info("No interactions yet")
    else:
        for h in hist[:40]:
            st.write(f"{h['time']} — **{h['household']}** — `{h['source']}`"); st.write(f"Q: {h['query']}"); st.code(h['advice'])
        if st.button("Download history JSON"):
            st.download_button("Download JSON", json.dumps(hist, indent=2), file_name="history.json", mime="application/json")

st.caption("This app demonstrates the multi-agent SAFER system. Replace simulated connectors with real transaction feeds and secure messaging gateways for production.")