/**
 * admin.js - SurgeGuard Admin Override Console
 *
 * Data strategy (works with AND without the FastAPI server):
 *   PRIMARY  - localStorage['surgeguard_flagged_txns']  (set by app.js simulation)
 *   SECONDARY - GET /api/v1/admin/flagged               (synced in background)
 *
 * Approval strategy:
 *   1. Update localStorage immediately so the UI reflects the change at once.
 *   2. POST /api/v1/admin/approve in background (fails silently if offline).
 */

// ── Shared localStorage key (must match app.js) ──────────────────────────────
const LS_KEY = 'surgeguard_flagged_txns';

function getLocalStore() {
  try { return JSON.parse(localStorage.getItem(LS_KEY) || '[]'); }
  catch (_) { return []; }
}
function saveLocalStore(entries) {
  try { localStorage.setItem(LS_KEY, JSON.stringify(entries)); }
  catch (_) {}
}

// ── State ─────────────────────────────────────────────────────────────────────
let allTransactions  = [];
let activeFilter     = 'all';
let searchQuery      = '';
let selectedTxId     = null;
let autoRefreshTimer = null;
const AUTO_REFRESH_MS = 4000;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const searchInput           = document.getElementById('searchInput');
const adminTableBody        = document.getElementById('adminTableBody');
const tableCountLabel       = document.getElementById('tableCountLabel');
const lastRefreshedLabel    = document.getElementById('lastRefreshedLabel');
const refreshBtn            = document.getElementById('refreshBtn');
const detailPanel           = document.getElementById('detailPanel');
const detailCloseBtn        = document.getElementById('detailCloseBtn');
const detailApproveBtn      = document.getElementById('detailApproveBtn');
const detailApproveBtnLabel = document.getElementById('detailApproveBtnLabel');
const detailApproveStatus   = document.getElementById('detailApproveStatus');
const detailMessage         = document.getElementById('detailMessage');
const serverStatusDot       = document.getElementById('serverStatusDot');
const serverStatusText      = document.getElementById('serverStatusText');
const statTotal    = document.getElementById('statTotal');
const statHalt     = document.getElementById('statHalt');
const statReview   = document.getElementById('statReview');
const statApproved = document.getElementById('statApproved');
const statPending  = document.getElementById('statPending');

// ── Helpers ───────────────────────────────────────────────────────────────────
function fmtTs(isoStr) {
  if (!isoStr) return '--';
  try {
    return new Date(isoStr).toLocaleString(undefined, {
      month: 'short', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
    });
  } catch (_) { return isoStr; }
}

function showDetailMsg(text, type) {
  detailMessage.textContent    = text;
  detailMessage.className      = 'admin-message ' + (type || 'error');
  detailMessage.style.display  = 'block';
}
function clearDetailMsg() { detailMessage.style.display = 'none'; }

function setServerStatus(online) {
  serverStatusDot.style.backgroundColor = online ? 'var(--color-approved)' : 'var(--color-halt)';
  serverStatusText.textContent = online ? 'ONLINE' : 'OFFLINE (local mode)';
  serverStatusText.style.color = online ? 'var(--color-approved)' : 'var(--color-halt)';
}

// ── Merge API data into localStorage (non-destructive) ───────────────────────
function mergeApiData(apiEntries) {
  const local = getLocalStore();
  let changed = false;

  apiEntries.forEach(apiTx => {
    const idx = local.findIndex(t => t.transaction_id === apiTx.transaction_id);
    if (idx === -1) {
      // New entry from API not yet in local
      local.unshift(apiTx);
      changed = true;
    } else {
      // If API says it was admin-approved but local doesn't know yet, sync it
      if (apiTx.approved_by_admin && !local[idx].approved_by_admin) {
        local[idx].approved_by_admin = true;
        local[idx].approved_at       = apiTx.approved_at;
        local[idx].current_decision  = 'APPROVE';
        changed = true;
      }
    }
  });

  if (changed) {
    // Sort newest first
    local.sort((a, b) => new Date(b.flagged_at) - new Date(a.flagged_at));
    if (local.length > 500) local.splice(500);
    saveLocalStore(local);
  }
  return local;
}

// ── Load data (localStorage first, then API merge) ───────────────────────────
async function loadTransactions() {
  // Step 1: render from localStorage immediately (instant, always works)
  allTransactions = getLocalStore();
  renderTable();
  updateStats();

  // Step 2: try to pull from API and merge in background
  try {
    const res = await fetch('/api/v1/admin/flagged?limit=200');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const apiData = await res.json();
    allTransactions = mergeApiData(apiData);
    setServerStatus(true);
  } catch (_) {
    setServerStatus(false);
    // Keep showing localStorage data — no change needed
  }

  renderTable();
  updateStats();
  lastRefreshedLabel.textContent = 'Refreshed: ' + new Date().toLocaleTimeString();
}

// ── Stats ─────────────────────────────────────────────────────────────────────
function updateStats() {
  const total    = allTransactions.length;
  const halt     = allTransactions.filter(t => t.original_decision === 'HALT').length;
  const review   = allTransactions.filter(t => t.original_decision === 'REVIEW').length;
  const approved = allTransactions.filter(t => t.approved_by_admin).length;
  const pending  = allTransactions.filter(t => !t.approved_by_admin).length;
  statTotal.textContent    = total;
  statHalt.textContent     = halt;
  statReview.textContent   = review;
  statApproved.textContent = approved;
  statPending.textContent  = pending;
}

// ── Filter + Search ───────────────────────────────────────────────────────────
function getFiltered() {
  let list = allTransactions;
  if (activeFilter === 'HALT')     list = list.filter(t => t.original_decision === 'HALT');
  else if (activeFilter === 'REVIEW')   list = list.filter(t => t.original_decision === 'REVIEW');
  else if (activeFilter === 'approved') list = list.filter(t => t.approved_by_admin);
  else if (activeFilter === 'pending')  list = list.filter(t => !t.approved_by_admin);

  if (searchQuery) {
    const q = searchQuery.toLowerCase();
    list = list.filter(t => t.transaction_id.toLowerCase().includes(q));
  }
  return list;
}

// ── Render Table ──────────────────────────────────────────────────────────────
function renderTable() {
  const rows = getFiltered();
  adminTableBody.innerHTML = '';

  if (rows.length === 0) {
    const msg = allTransactions.length === 0
      ? 'No flagged transactions yet. Open the live dashboard, switch to "Coordinated Attack" scenario, and wait a few seconds.'
      : 'No transactions match the current filter or search.';
    adminTableBody.innerHTML = `
      <tr><td colspan="10" class="admin-empty-cell">
        <div class="admin-empty-state"><span>${msg}</span></div>
      </td></tr>`;
    tableCountLabel.textContent = '0 transactions';
    return;
  }

  tableCountLabel.textContent = rows.length + ' transaction' + (rows.length !== 1 ? 's' : '');

  rows.forEach(tx => {
    const approved       = tx.approved_by_admin;
    const curDecision    = approved ? 'APPROVE' : tx.current_decision;
    const isSelected     = tx.transaction_id === selectedTxId;

    const tr = document.createElement('tr');
    tr.className      = isSelected ? 'selected-row' : '';
    tr.dataset.txid   = tx.transaction_id;
    tr.style.cursor   = 'pointer';

    tr.innerHTML = `
      <td><button class="row-selector-btn" data-txid="${tx.transaction_id}" title="View details">&#9654;</button></td>
      <td style="font-family:var(--font-mono);color:var(--text-primary);font-weight:500;">${tx.transaction_id}</td>
      <td>${tx.amount || '--'}</td>
      <td><span class="ledger-badge ${tx.original_decision.toLowerCase()}">${tx.original_decision}</span></td>
      <td><span class="ledger-badge ${curDecision.toLowerCase()}">${curDecision}</span></td>
      <td>${parseFloat(tx.ml_fraud_prob).toFixed(4)}</td>
      <td>${parseFloat(tx.anomaly_score).toFixed(4)}</td>
      <td style="font-weight:600;color:var(--text-primary);">${parseFloat(tx.risk_score).toFixed(4)}</td>
      <td style="white-space:nowrap;">${fmtTs(tx.flagged_at)}</td>
      <td>${approved
        ? '<span class="btn-row-approved">&#10003; Approved</span>'
        : `<button class="btn-row-approve" data-txid="${tx.transaction_id}">Approve</button>`
      }</td>`;

    adminTableBody.appendChild(tr);
  });

  // Event: row click -> open detail
  adminTableBody.querySelectorAll('tr[data-txid]').forEach(row => {
    row.addEventListener('click', e => {
      if (e.target.tagName === 'BUTTON') return;
      openDetail(row.dataset.txid);
    });
  });
  adminTableBody.querySelectorAll('.row-selector-btn').forEach(btn => {
    btn.addEventListener('click', () => openDetail(btn.dataset.txid));
  });
  // Event: inline approve button
  adminTableBody.querySelectorAll('.btn-row-approve').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      doApprove(btn.dataset.txid, () => {
        allTransactions = getLocalStore();
        renderTable();
        updateStats();
        if (selectedTxId === btn.dataset.txid) openDetail(btn.dataset.txid);
      });
    });
  });
}

// ── Detail Panel ──────────────────────────────────────────────────────────────
function openDetail(txId) {
  const tx = allTransactions.find(t => t.transaction_id === txId);
  if (!tx) return;
  selectedTxId = txId;
  clearDetailMsg();

  adminTableBody.querySelectorAll('tr').forEach(r => r.classList.remove('selected-row'));
  const row = adminTableBody.querySelector(`tr[data-txid="${txId}"]`);
  if (row) row.classList.add('selected-row');

  document.getElementById('detailTxId').textContent = tx.transaction_id;

  const cur   = tx.approved_by_admin ? 'approve' : tx.current_decision.toLowerCase();
  const label = tx.approved_by_admin ? 'APPROVED' : tx.current_decision;
  const badge = document.getElementById('detailCurrentBadge');
  badge.textContent = label;
  badge.className   = 'decision-badge ' + cur;

  document.getElementById('detailOriginalDecision').textContent = tx.original_decision;
  document.getElementById('detailAmount').textContent           = tx.amount || '--';
  document.getElementById('detailFlaggedAt').textContent        = fmtTs(tx.flagged_at);
  document.getElementById('detailML').textContent               = parseFloat(tx.ml_fraud_prob).toFixed(4);
  document.getElementById('detailAnomaly').textContent          = parseFloat(tx.anomaly_score).toFixed(4);
  document.getElementById('detailRisk').textContent             = parseFloat(tx.risk_score).toFixed(4);
  document.getElementById('detailVelocity').textContent         = parseFloat(tx.velocity_surge_index).toFixed(4);

  const approvedEl = document.getElementById('detailApprovedStatus');
  if (tx.approved_by_admin) {
    approvedEl.textContent = 'Yes — ' + fmtTs(tx.approved_at);
    approvedEl.style.color = 'var(--color-approved)';
  } else {
    approvedEl.textContent = 'No';
    approvedEl.style.color = '';
  }

  const list = document.getElementById('detailReasonsList');
  list.innerHTML = '';
  (tx.reasons || []).forEach(r => {
    const li = document.createElement('li');
    li.textContent = r;
    list.appendChild(li);
  });

  if (tx.approved_by_admin) {
    detailApproveBtn.disabled = true;
    detailApproveBtnLabel.textContent = 'Already Approved';
  } else {
    detailApproveBtn.disabled = false;
    detailApproveBtnLabel.textContent = 'Approve Transaction';
    detailApproveBtn.onclick = () => doApprove(txId, () => {
      allTransactions = getLocalStore();
      renderTable();
      updateStats();
      openDetail(txId);
    });
  }
  detailApproveStatus.textContent = '';
  detailPanel.style.display = 'block';
  detailPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function closeDetail() {
  detailPanel.style.display = 'none';
  selectedTxId = null;
  adminTableBody.querySelectorAll('tr').forEach(r => r.classList.remove('selected-row'));
}

// ── Approve (localStorage first, then API) ────────────────────────────────────
function doApprove(txId, onDone) {
  // 1. Update localStorage immediately
  const store = getLocalStore();
  const idx   = store.findIndex(t => t.transaction_id === txId);
  if (idx !== -1 && !store[idx].approved_by_admin) {
    const now = new Date().toISOString();
    store[idx].approved_by_admin  = true;
    store[idx].approved_at        = now;
    store[idx].current_decision   = 'APPROVE';
    saveLocalStore(store);
  }

  detailApproveBtn.disabled = true;
  detailApproveBtnLabel.textContent = 'Approved!';
  detailApproveStatus.textContent   = 'Override recorded at ' + new Date().toLocaleTimeString();
  showDetailMsg('Transaction approved. The override is stored locally and will sync to the server when online.', 'info');

  if (onDone) onDone();

  // 2. Background sync to API
  fetch('/api/v1/admin/approve/' + encodeURIComponent(txId), { method: 'POST' })
    .catch(() => {}); // silent if offline
}

// ── Auto-refresh ──────────────────────────────────────────────────────────────
function startAutoRefresh() {
  if (autoRefreshTimer) clearInterval(autoRefreshTimer);
  autoRefreshTimer = setInterval(loadTransactions, AUTO_REFRESH_MS);
}

// ── Event Wiring ──────────────────────────────────────────────────────────────
searchInput.addEventListener('input', () => {
  searchQuery = searchInput.value.trim();
  renderTable();
});

document.querySelectorAll('.filter-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    activeFilter = tab.dataset.filter;
    renderTable();
  });
});

refreshBtn.addEventListener('click', loadTransactions);
detailCloseBtn.addEventListener('click', closeDetail);

// ── Boot ──────────────────────────────────────────────────────────────────────
loadTransactions();
startAutoRefresh();
