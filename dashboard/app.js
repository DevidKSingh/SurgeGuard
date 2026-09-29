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

// Update UI with new transaction
function processNextTransaction() {
  if (isPaused) return;

  const tx = generateTransaction(currentScenario);
  txCount++;

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
