/**
 * Securing the Surge - Real-Time Dashboard Controller
 * Handles scenario playback, canvas rendering, risk decomposition, and API calls.
 */

// Global State
let currentScenario = 'normal'; // 'normal' | 'flash_sale' | 'bot_attack'
let isPaused = false;
let txCount = 0;
let approvedCount = 4281;
let reviewCount = 12;
let haltedCount = 38;
let savedAmount = 42910;
let currentTps = 18.4;

// Historical series for live canvas chart (last 60 ticks)
const maxHistory = 50;
const historyTps = Array(maxHistory).fill(18.0);
const historyRisk = Array(maxHistory).fill(0.04);

// DOM Elements
const engineStatusBadge = document.getElementById('engineStatusBadge');
const kpiTpsValue = document.getElementById('kpiTpsValue');
const kpiTpsBar = document.getElementById('kpiTpsBar');
const kpiTpsMeta = document.getElementById('kpiTpsMeta');
const kpiApprovedCount = document.getElementById('kpiApprovedCount');
const kpiReviewCount = document.getElementById('kpiReviewCount');
const kpiHaltedCount = document.getElementById('kpiHaltedCount');
const kpiProtectedAmount = document.getElementById('kpiProtectedAmount');

const sigMlValue = document.getElementById('sigMlValue');
const sigMlBar = document.getElementById('sigMlBar');
const sigIsoValue = document.getElementById('sigIsoValue');
const sigIsoBar = document.getElementById('sigIsoBar');
const sigBurstValue = document.getElementById('sigBurstValue');
const sigBurstBar = document.getElementById('sigBurstBar');

const gaugeNumber = document.getElementById('gaugeNumber');
const gaugeDecisionBadge = document.getElementById('gaugeDecisionBadge');
const transactionTableBody = document.getElementById('transactionTableBody');
const canvas = document.getElementById('velocityCanvas');
const ctx = canvas.getContext('2d');

// Scenario Switching
const btnNormal = document.getElementById('btnScenarioNormal');
const btnFlashSale = document.getElementById('btnScenarioFlashSale');
const btnBotAttack = document.getElementById('btnScenarioBotAttack');

function setScenario(sc) {
  currentScenario = sc;
  btnNormal.classList.toggle('active', sc === 'normal');
  btnFlashSale.classList.toggle('active', sc === 'flash_sale');
  btnBotAttack.classList.toggle('active', sc === 'bot_attack');
}

btnNormal.addEventListener('click', () => setScenario('normal'));
btnFlashSale.addEventListener('click', () => setScenario('flash_sale'));
btnBotAttack.addEventListener('click', () => setScenario('bot_attack'));

document.getElementById('btnPauseStream').addEventListener('click', function() {
  isPaused = !isPaused;
  this.innerText = isPaused ? 'Resume Feed' : 'Pause Feed';
});

document.getElementById('btnClearLedger').addEventListener('click', () => {
  transactionTableBody.innerHTML = '';
});

// Canvas Live Chart Render
function drawLiveChart() {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // Subtle grid lines
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  for (let y = 30; y < h; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Draw TPS line (Subdued steel cyan)
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 2.0;
  ctx.beginPath();
  const maxScaleTps = 200;
  for (let i = 0; i < historyTps.length; i++) {
    const x = (i / (maxHistory - 1)) * w;
    const y = h - (historyTps[i] / maxScaleTps) * (h - 20) - 10;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Draw Risk line (Subdued crimson red)
  ctx.strokeStyle = '#ef4444';
  ctx.lineWidth = 2.0;
  ctx.beginPath();
  for (let i = 0; i < historyRisk.length; i++) {
    const x = (i / (maxHistory - 1)) * w;
    const y = h - historyRisk[i] * (h - 20) - 10;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// Generate Realistic Simulated Transaction Based on Scenario
function generateTransaction(scenario) {
  const now = Date.now() / 1000;
  let amount = 0;
  let mlProb = 0;
  let anomaly = 0;
  let velocityFactor = 0;
  let decision = 'APPROVE';
  let rationale = '';

  if (scenario === 'normal') {
    amount = (Math.random() * 85 + 5).toFixed(2);
    mlProb = Math.random() * 0.05 + 0.01;
    anomaly = Math.random() * 0.12 + 0.02;
    velocityFactor = Math.random() * 0.2 + 0.1;
    currentTps = (15 + Math.random() * 8).toFixed(1);
    decision = 'APPROVE';
    rationale = 'Nominal spending pattern • Verified user baseline';
  } else if (scenario === 'flash_sale') {
    // Flash Sale: Unusually high velocity & burst, but legitimate feature signatures
    amount = (Math.random() * 120 + 20).toFixed(2);
    mlProb = Math.random() * 0.08 + 0.01;
    anomaly = Math.random() * 0.14 + 0.03; // Low anomaly!
    velocityFactor = Math.random() * 0.3 + 0.85; // High burst!
    currentTps = (130 + Math.random() * 45).toFixed(1);
    decision = 'APPROVE';
    rationale = 'Flash-sale burst verified • Normal feature distribution • Approved';
  } else if (scenario === 'bot_attack') {
    // Bot Attack: High velocity AND high anomaly / fraud indicators
    const isBotTx = Math.random() > 0.25;
    if (isBotTx) {
      amount = (Math.random() * 250 + 15).toFixed(2);
      mlProb = Math.random() * 0.45 + 0.52;
      anomaly = Math.random() * 0.35 + 0.65;
      velocityFactor = Math.random() * 0.25 + 0.80;
      currentTps = (165 + Math.random() * 50).toFixed(1);
      
      const composite = (0.75 * mlProb + 0.15 * anomaly + 0.10 * (velocityFactor * anomaly));
      if (composite >= 0.70) {
        decision = 'HALT';
        rationale = 'Distributed bot draining signature isolated • Quarantined';
      } else {
        decision = 'REVIEW';
        rationale = 'Suspicious velocity burst + atypical feature vector • Step-up challenge';
      }
    } else {
      // Interleaved normal shopper during the surge
      amount = (Math.random() * 60 + 10).toFixed(2);
      mlProb = 0.03;
      anomaly = 0.07;
      velocityFactor = 0.9;
      currentTps = 150;
      decision = 'APPROVE';
      rationale = 'Concurrent legitimate shopper successfully approved';
    }
  }

  const compositeRisk = Math.min(1.0, (0.75 * mlProb + 0.15 * anomaly + 0.10 * (velocityFactor * anomaly)));

  return {
    id: `TX-${Math.floor(now % 100000)}-${Math.floor(Math.random() * 8999 + 1000)}`,
    time: new Date().toLocaleTimeString(),
    amount: `$${amount}`,
    rawAmount: parseFloat(amount),
    mlProb: mlProb.toFixed(3),
    anomaly: anomaly.toFixed(3),
    velocity: velocityFactor.toFixed(3),
    risk: compositeRisk.toFixed(3),
    decision: decision,
    rationale: rationale
  };
}

// ── Shared localStorage store (works with or without the server) ─────────────
const LS_KEY = 'surgeguard_flagged_txns';

function getLocalFlaggedStore() {
  try {
    return JSON.parse(localStorage.getItem(LS_KEY) || '[]');
  } catch (_) { return []; }
}

function saveLocalFlaggedStore(entries) {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(entries));
  } catch (_) { /* quota exceeded - ignore */ }
}

/**
 * Save a HALT or REVIEW transaction to localStorage immediately (so admin page
 * can display it without requiring the server), then also try to sync to the API.
 */
function saveFlaggedTransaction(tx) {
  // 1. Write to localStorage right away
  const store = getLocalFlaggedStore();
  const exists = store.some(t => t.transaction_id === tx.id);
  if (!exists) {
    store.unshift({
      transaction_id:      tx.id,
      original_decision:   tx.decision,
      current_decision:    tx.decision,
      risk_score:          parseFloat(tx.risk),
      ml_fraud_prob:       parseFloat(tx.mlProb),
      anomaly_score:       parseFloat(tx.anomaly),
      velocity_surge_index: parseFloat(tx.velocity),
      amount:              tx.amount,
      reasons:             [tx.rationale],
      flagged_at:          new Date().toISOString(),
      approved_by_admin:   false,
      approved_at:         null
    });
    if (store.length > 500) store.splice(500); // cap size
    saveLocalFlaggedStore(store);
  }

  // 2. Background sync to API (fire-and-forget; never blocks the simulation)
  fetch('/api/v1/admin/register', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      transaction_id:      tx.id,
      decision:            tx.decision,
      risk_score:          parseFloat(tx.risk),
      ml_fraud_prob:       parseFloat(tx.mlProb),
      anomaly_score:       parseFloat(tx.anomaly),
      velocity_surge_index: parseFloat(tx.velocity),
      amount:              tx.amount,
      reasons:             [tx.rationale]
    })
  }).catch(() => {}); // silently ignore if server is offline
}

// Update UI with new transaction
function processNextTransaction() {
  if (isPaused) return;

  const tx = generateTransaction(currentScenario);
  txCount++;

  // Persist flagged transactions so admin page can find them
  if (tx.decision === 'HALT' || tx.decision === 'REVIEW') {
    saveFlaggedTransaction(tx);
  }

  // Update counters
  if (tx.decision === 'APPROVE') approvedCount++;
  else if (tx.decision === 'REVIEW') reviewCount++;
  else if (tx.decision === 'HALT') {
    haltedCount++;
    savedAmount += tx.rawAmount;
  }

  // Update KPIs
  kpiTpsValue.innerText = currentTps;
  const pctTps = Math.min(100, (currentTps / 200) * 100);
  kpiTpsBar.style.width = `${pctTps}%`;
  kpiApprovedCount.innerText = approvedCount.toLocaleString();
  kpiReviewCount.innerText = reviewCount.toLocaleString();
  kpiHaltedCount.innerText = haltedCount.toLocaleString();
  kpiProtectedAmount.innerText = `$${Math.round(savedAmount).toLocaleString()} saved in fund drainage`;

  if (currentScenario === 'normal') {
    kpiTpsMeta.innerText = 'Within nominal traffic envelope';
  } else if (currentScenario === 'flash_sale') {
    kpiTpsMeta.innerText = 'Flash-sale surge active (Traffic x7.5 baseline)';
  } else {
    kpiTpsMeta.innerText = 'Bot attack blitz detected • Quarantine engaged';
  }

  // Update Tri-Signal Bars
  sigMlValue.innerText = tx.mlProb;
  sigMlBar.style.width = `${Math.min(100, tx.mlProb * 100)}%`;

  sigIsoValue.innerText = tx.anomaly;
  sigIsoBar.style.width = `${Math.min(100, tx.anomaly * 100)}%`;

  sigBurstValue.innerText = tx.velocity;
  sigBurstBar.style.width = `${Math.min(100, tx.velocity * 100)}%`;

  // Update Composite Gauge
  gaugeNumber.innerText = tx.risk;
  gaugeDecisionBadge.className = `decision-badge ${tx.decision.toLowerCase()}`;
  gaugeDecisionBadge.innerText = tx.decision;

  // Append row to ledger table
  const rationaleColorClass = tx.decision === 'HALT' ? 'text-rose' : (tx.decision === 'REVIEW' ? 'text-amber' : '');
  const row = document.createElement('tr');
  row.innerHTML = `
    <td>${tx.id}</td>
    <td>${tx.time}</td>
    <td>${tx.amount}</td>
    <td>${tx.mlProb}</td>
    <td>${tx.anomaly}</td>
    <td>${tx.velocity}</td>
    <td style="font-weight:600; color:var(--text-primary);">${tx.risk}</td>
    <td><span class="ledger-badge ${tx.decision.toLowerCase()}">${tx.decision}</span></td>
    <td class="${rationaleColorClass}">${tx.rationale}</td>
  `;

  transactionTableBody.insertBefore(row, transactionTableBody.firstChild);
  if (transactionTableBody.children.length > 25) {
    transactionTableBody.removeChild(transactionTableBody.lastChild);
  }

  // Update chart series
  historyTps.shift();
  historyTps.push(parseFloat(currentTps));
  historyRisk.shift();
  historyRisk.push(parseFloat(tx.risk));

  drawLiveChart();
}

// Dynamic Loop interval based on scenario
function runSimulationLoop() {
  processNextTransaction();
  let delay = 650;
  if (currentScenario === 'flash_sale') delay = 180;
  if (currentScenario === 'bot_attack') delay = 220;
  setTimeout(runSimulationLoop, delay);
}

// Initial Kickoff
drawLiveChart();
runSimulationLoop();

// =====================================================================
// Admin Override Panel Controller
// =====================================================================

const API_BASE = window.location.origin;
let adminOverrideCount = 0;

const adminTxIdInput    = document.getElementById('adminTxIdInput');
const adminSearchBtn    = document.getElementById('adminSearchBtn');
const adminSearchLabel  = document.getElementById('adminSearchBtnLabel');
const adminResultCard   = document.getElementById('adminResultCard');
const adminMessage      = document.getElementById('adminMessage');
const adminApproveBtn   = document.getElementById('adminApproveBtn');
const adminApproveBtnLabel = document.getElementById('adminApproveBtnLabel');
const adminApproveStatus   = document.getElementById('adminApproveStatus');
const adminOverrideCountBadge = document.getElementById('adminOverrideCount');

// Show / hide the inline message banner
function showAdminMessage(text, type = 'error') {
  adminMessage.textContent = text;
  adminMessage.className = `admin-message ${type}`;
  adminMessage.style.display = 'block';
}
function clearAdminMessage() {
  adminMessage.style.display = 'none';
  adminMessage.textContent = '';
}

// Render the result card with flagged transaction data
function renderResultCard(tx) {
  clearAdminMessage();

  document.getElementById('adminResultTxId').textContent = tx.transaction_id;

  const badge = document.getElementById('adminResultCurrentBadge');
  badge.textContent = tx.current_decision;
  badge.className = `decision-badge ${tx.current_decision.toLowerCase()}`;

  document.getElementById('adminResultOriginalDecision').textContent = tx.original_decision;

  document.getElementById('adminResML').textContent       = tx.ml_fraud_prob.toFixed(4);
  document.getElementById('adminResAnomaly').textContent  = tx.anomaly_score.toFixed(4);
  document.getElementById('adminResRisk').textContent     = tx.risk_score.toFixed(4);
  document.getElementById('adminResVelocity').textContent = tx.velocity_surge_index.toFixed(4);

  // Format timestamp readably
  const flaggedDate = new Date(tx.flagged_at);
  document.getElementById('adminResFlaggedAt').textContent = flaggedDate.toUTCString().replace('GMT', 'UTC');

  const approvedStatusEl = document.getElementById('adminResApprovedStatus');
  if (tx.approved_by_admin) {
    approvedStatusEl.textContent = `Yes — ${new Date(tx.approved_at).toUTCString().replace('GMT', 'UTC')}`;
    approvedStatusEl.style.color = 'var(--color-approved)';
  } else {
    approvedStatusEl.textContent = 'No';
    approvedStatusEl.style.color = '';
  }

  // Populate ML rationale reasons
  const list = document.getElementById('adminResultReasonsList');
  list.innerHTML = '';
  (tx.reasons || []).forEach(r => {
    const li = document.createElement('li');
    li.textContent = r;
    list.appendChild(li);
  });

  // Toggle the approve button state
  if (tx.approved_by_admin) {
    adminApproveBtn.disabled = true;
    adminApproveBtnLabel.textContent = 'Already Approved';
    adminApproveStatus.textContent = '';
  } else {
    adminApproveBtn.disabled = false;
    adminApproveBtnLabel.textContent = 'Approve Transaction';
    adminApproveStatus.textContent = '';
    // Attach the approval handler (replace previous one)
    adminApproveBtn.onclick = () => handleAdminApprove(tx.transaction_id);
  }

  adminResultCard.style.display = 'block';
}

// Search handler
async function handleAdminSearch() {
  const txId = adminTxIdInput.value.trim();
  if (!txId) {
    showAdminMessage('Please enter a Transaction ID to search.', 'error');
    return;
  }

  adminResultCard.style.display = 'none';
  clearAdminMessage();
  adminSearchBtn.disabled = true;
  adminSearchLabel.textContent = 'Searching…';

  try {
    const res = await fetch(`${API_BASE}/api/v1/admin/transaction/${encodeURIComponent(txId)}`);
    if (res.status === 404) {
      showAdminMessage(
        `Transaction "${txId}" not found. Only HALT or REVIEW transactions are stored. ` +
        'Make sure the ID is exact (the feed must be running against the live API).',
        'error'
      );
    } else if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      showAdminMessage(`API error ${res.status}: ${err.detail || res.statusText}`, 'error');
    } else {
      const tx = await res.json();
      renderResultCard(tx);
    }
  } catch (e) {
    showAdminMessage(
      'Could not reach the SurgeGuard API. Make sure the server is running (python run_server.py).',
      'error'
    );
  } finally {
    adminSearchBtn.disabled = false;
    adminSearchLabel.textContent = 'Search';
  }
}

// Approve handler
async function handleAdminApprove(txId) {
  adminApproveBtn.disabled = true;
  adminApproveBtnLabel.textContent = 'Approving…';
  adminApproveStatus.textContent = '';

  try {
    const res = await fetch(`${API_BASE}/api/v1/admin/approve/${encodeURIComponent(txId)}`, {
      method: 'POST',
    });

    if (res.status === 409) {
      // Already approved
      const err = await res.json();
      adminApproveStatus.textContent = err.detail || 'Already approved.';
      adminApproveBtnLabel.textContent = 'Already Approved';
    } else if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      showAdminMessage(`Approve failed: ${err.detail || res.statusText}`, 'error');
      adminApproveBtn.disabled = false;
      adminApproveBtnLabel.textContent = 'Approve Transaction';
    } else {
      const result = await res.json();

      // Update counter
      adminOverrideCount++;
      adminOverrideCountBadge.textContent = `${adminOverrideCount} override${adminOverrideCount !== 1 ? 's' : ''} issued`;

      // Update badge in result card to APPROVE
      const badge = document.getElementById('adminResultCurrentBadge');
      badge.textContent = 'APPROVE';
      badge.className = 'decision-badge approve';

      const approvedStatusEl = document.getElementById('adminResApprovedStatus');
      approvedStatusEl.textContent = `Yes — ${new Date(result.approved_at).toUTCString().replace('GMT', 'UTC')}`;
      approvedStatusEl.style.color = 'var(--color-approved)';

      adminApproveBtnLabel.textContent = 'Approved ✓';
      adminApproveStatus.textContent = `Override recorded at ${new Date(result.approved_at).toLocaleTimeString()}`;

      showAdminMessage(`✓ ${result.message}`, 'info');
    }
  } catch (e) {
    showAdminMessage('Network error during approval. Is the server running?', 'error');
    adminApproveBtn.disabled = false;
    adminApproveBtnLabel.textContent = 'Approve Transaction';
  }
}

// Wire up events
adminSearchBtn.addEventListener('click', handleAdminSearch);
adminTxIdInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') handleAdminSearch();
});
