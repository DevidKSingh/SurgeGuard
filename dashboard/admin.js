/**
 * admin.js - SurgeGuard Fraud Operations Center & Model Lifecycle Controller
 *
 * Data Strategy:
 *   PRIMARY   - localStorage['surgeguard_flagged_txns'] (instant local rendering)
 *   SECONDARY - GET /api/v1/admin/flagged              (synced periodically in background)
 *
 * Human-in-the-Loop Feedback:
 *   POST /api/v1/admin/feedback/{tx_id}                (persisted to SQLite feedback_records)
 *
 * Model Management & Retraining:
 *   GET /api/v1/admin/model-status                     (active champion metrics & feedback counts)
 *   POST /api/v1/admin/retrain                         (challenger training, validation & promotion)
 */

// ── Shared localStorage key ──────────────────────────────────────────────────
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
let allTransactions       = [];
let activeFilter          = 'all';
let searchQuery           = '';
let selectedTxId          = null;
let selectedFeedbackLabel = null;
let autoRefreshTimer      = null;
let currentModelStatus    = null;
let activeView            = 'queue'; // 'queue' | 'model'
const AUTO_REFRESH_MS     = 4000;

// ── DOM References ────────────────────────────────────────────────────────────
// View Switcher
const tabViewQueue             = document.getElementById('tabViewQueue');
const tabViewModel             = document.getElementById('tabViewModel');
const sectionQueueView         = document.getElementById('sectionQueueView');
const sectionModelCenterView   = document.getElementById('sectionModelCenterView');
const tabQueueBadge            = document.getElementById('tabQueueBadge');

// Filter & Search
const searchInput              = document.getElementById('searchInput');
const adminTableBody           = document.getElementById('adminTableBody');
const tableCountLabel          = document.getElementById('tableCountLabel');
const lastRefreshedLabel       = document.getElementById('lastRefreshedLabel');
const refreshBtn               = document.getElementById('refreshBtn');

// Detail Panel
const detailPanel              = document.getElementById('detailPanel');
const detailCloseBtn           = document.getElementById('detailCloseBtn');
const detailApproveBtn         = document.getElementById('detailApproveBtn');
const detailApproveBtnLabel    = document.getElementById('detailApproveBtnLabel');
const detailApproveStatus      = document.getElementById('detailApproveStatus');
const detailMessage            = document.getElementById('detailMessage');
const serverStatusDot          = document.getElementById('serverStatusDot');
const serverStatusText         = document.getElementById('serverStatusText');

// Executive Stats
const statTotal                = document.getElementById('statTotal');
const statTotalSub             = document.getElementById('statTotalSub');
const statHalt                 = document.getElementById('statHalt');
const statHaltSub              = document.getElementById('statHaltSub');
const statReview               = document.getElementById('statReview');
const statReviewSub            = document.getElementById('statReviewSub');
const statApproved             = document.getElementById('statApproved');
const statApprovedSub          = document.getElementById('statApprovedSub');
const statPending              = document.getElementById('statPending');
const statPendingSub           = document.getElementById('statPendingSub');

// Risk Distribution Bar
const barApproveSeg            = document.getElementById('barApproveSeg');
const barReviewSeg             = document.getElementById('barReviewSeg');
const barHaltSeg               = document.getElementById('barHaltSeg');
const riskApproveCount         = document.getElementById('riskApproveCount');
const riskReviewCount          = document.getElementById('riskReviewCount');
const riskHaltCount            = document.getElementById('riskHaltCount');
const riskApprovePct           = document.getElementById('riskApprovePct');
const riskReviewPct            = document.getElementById('riskReviewPct');
const riskHaltPct              = document.getElementById('riskHaltPct');

// Top Telemetry Badges
const activeModelName          = document.getElementById('activeModelName');
const feedbackCountLabel       = document.getElementById('feedbackCountLabel');
const feedbackProgressTag      = document.getElementById('feedbackProgressTag');
const btnRetrainModal          = document.getElementById('btnRetrainModal');
const btnManualTriggerRetrain  = document.getElementById('btnManualTriggerRetrain');

// Feedback Form in Detail Panel
const btnLabelLegit            = document.getElementById('btnLabelLegit');
const btnLabelFraud            = document.getElementById('btnLabelFraud');
const feedbackNoteInput        = document.getElementById('feedbackNoteInput');
const btnSubmitFeedback        = document.getElementById('btnSubmitFeedback');
const btnSubmitFeedbackLabel   = document.getElementById('btnSubmitFeedbackLabel');
const feedbackSubmitMsg        = document.getElementById('feedbackSubmitMsg');
const feedbackStatusBadge      = document.getElementById('feedbackStatusBadge');
const feedbackRecordedAt       = document.getElementById('feedbackRecordedAt');

// Model Center Elements
const centerActiveModelVersion = document.getElementById('centerActiveModelVersion');
const centerActivePrAuc        = document.getElementById('centerActivePrAuc');
const centerActiveRocAuc       = document.getElementById('centerActiveRocAuc');
const centerActivatedAt        = document.getElementById('centerActivatedAt');
const centerFeedbackCountLabel = document.getElementById('centerFeedbackCountLabel');
const centerChallengerVersion  = document.getElementById('centerChallengerVersion');
const centerPromotionOutcomeBadge = document.getElementById('centerPromotionOutcomeBadge');
const centerChampPr            = document.getElementById('centerChampPr');
const centerChalPr             = document.getElementById('centerChalPr');
const centerDeltaPr            = document.getElementById('centerDeltaPr');
const centerChampRoc           = document.getElementById('centerChampRoc');
const centerChalRoc            = document.getElementById('centerChalRoc');
const centerDeltaRoc           = document.getElementById('centerDeltaRoc');
const centerOutcomeReason      = document.getElementById('centerOutcomeReason');
const modelHistoryTableBody    = document.getElementById('modelHistoryTableBody');

// Retraining Modal Elements
const retrainModal             = document.getElementById('retrainModal');
const closeRetrainModalBtn     = document.getElementById('closeRetrainModalBtn');
const modalActionBtn           = document.getElementById('modalActionBtn');
const retrainLoadingState      = document.getElementById('retrainLoadingState');
const retrainResultsState      = document.getElementById('retrainResultsState');
const retrainProgressText      = document.getElementById('retrainProgressText');
const modalPromotionBadge      = document.getElementById('modalPromotionBadge');
const modalChampionName        = document.getElementById('modalChampionName');
const modalChallengerName      = document.getElementById('modalChallengerName');
const modalFeedbackUsed        = document.getElementById('modalFeedbackUsed');
const rowChampPr               = document.getElementById('rowChampPr');
const rowChalPr                = document.getElementById('rowChalPr');
const rowDeltaPr               = document.getElementById('rowDeltaPr');
const rowChampRoc              = document.getElementById('rowChampRoc');
const rowChalRoc               = document.getElementById('rowChalRoc');
const rowDeltaRoc              = document.getElementById('rowDeltaRoc');
const modalReasonText          = document.getElementById('modalReasonText');

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

function fmtTimeOnly(isoStr) {
  if (!isoStr) return '--';
  try {
    const d = new Date(isoStr);
    return d.toTimeString().split(' ')[0];
  } catch (_) { return isoStr; }
}

function showDetailMsg(text, type) {
  if (!detailMessage) return;
  detailMessage.textContent    = text;
  detailMessage.className      = 'admin-message ' + (type || 'error');
  detailMessage.style.display  = 'block';
}
function clearDetailMsg() {
  if (detailMessage) detailMessage.style.display = 'none';
}

function setServerStatus(online) {
  if (serverStatusDot) {
    serverStatusDot.style.backgroundColor = online ? 'var(--color-approved)' : 'var(--color-halt)';
  }
  if (serverStatusText) {
    serverStatusText.textContent = online ? 'ONLINE' : 'LOCAL MODE';
    serverStatusText.style.color = online ? 'var(--text-approved)' : 'var(--text-halt)';
  }
}

// ── View Switching Logic ──────────────────────────────────────────────────────
function switchView(viewName) {
  activeView = viewName;
  if (viewName === 'queue') {
    if (tabViewQueue) tabViewQueue.classList.add('active');
    if (tabViewModel) tabViewModel.classList.remove('active');
    if (sectionQueueView) sectionQueueView.style.display = 'block';
    if (sectionModelCenterView) sectionModelCenterView.style.display = 'none';
  } else {
    if (tabViewModel) tabViewModel.classList.add('active');
    if (tabViewQueue) tabViewQueue.classList.remove('active');
    if (sectionQueueView) sectionQueueView.style.display = 'none';
    if (sectionModelCenterView) sectionModelCenterView.style.display = 'flex';
    renderModelCenter();
  }
}

if (tabViewQueue) tabViewQueue.addEventListener('click', () => switchView('queue'));
if (tabViewModel) tabViewModel.addEventListener('click', () => switchView('model'));

// ── Load Model Status & Model Center ──────────────────────────────────────────
async function loadModelStatus() {
  try {
    const res = await fetch('/api/v1/admin/model-status');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    currentModelStatus = data;

    // Header updates
    if (activeModelName) activeModelName.textContent = data.active_model_version || 'model_v2';
    const totalFb = data.feedback_stats ? data.feedback_stats.total_verified : 0;
    const thresh  = data.retrain_threshold || 10;
    if (feedbackCountLabel) feedbackCountLabel.textContent = `${totalFb}`;
    if (feedbackProgressTag) {
      feedbackProgressTag.title = `${totalFb} verified samples recorded (Retrain target: ${thresh})`;
    }

    if (btnRetrainModal) {
      if (data.ready_for_retraining) {
        btnRetrainModal.style.borderColor = 'var(--color-system)';
        btnRetrainModal.style.boxShadow   = '0 0 10px rgba(99, 102, 241, 0.4)';
        btnRetrainModal.title = `Retrain threshold reached (${totalFb}/${thresh})! Ready to train challenger model.`;
      } else {
        btnRetrainModal.style.borderColor = '';
        btnRetrainModal.style.boxShadow   = '';
      }
    }

    renderModelCenter();
  } catch (_) {
    // Local / offline mode
  }
}

// ── Render Model Center & Registry Table ──────────────────────────────────────
function renderModelCenter() {
  if (!currentModelStatus) return;

  const data = currentModelStatus;
  const champMetrics = data.champion_metrics || { pr_auc: 0.4804, roc_auc: 0.7937 };

  // 1. Active Champion Card
  if (centerActiveModelVersion) centerActiveModelVersion.textContent = data.active_model_version || 'model_v2';
  if (centerActivePrAuc)  centerActivePrAuc.textContent  = champMetrics.pr_auc ? champMetrics.pr_auc.toFixed(4) : '--';
  if (centerActiveRocAuc) centerActiveRocAuc.textContent = champMetrics.roc_auc ? champMetrics.roc_auc.toFixed(4) : '--';
  if (centerActivatedAt)  centerActivatedAt.textContent  = fmtTs(data.activated_at);
  if (centerFeedbackCountLabel) {
    const fb = data.feedback_stats ? data.feedback_stats.total_verified : 0;
    centerFeedbackCountLabel.textContent = `${fb} samples`;
  }

  // 2. Last Challenger Evaluation Card
  const history = data.history || [];
  // Find the most recent evaluation entry that tested a challenger
  const lastEval = [...history].reverse().find(h => h.challenger_version);

  if (lastEval) {
    if (centerChallengerVersion) centerChallengerVersion.textContent = lastEval.challenger_version;

    if (centerPromotionOutcomeBadge) {
      if (lastEval.promoted) {
        centerPromotionOutcomeBadge.className   = 'feedback-pill legit';
        centerPromotionOutcomeBadge.textContent = 'PROMOTED TO CHAMPION';
      } else {
        centerPromotionOutcomeBadge.className   = 'feedback-pill fraud';
        centerPromotionOutcomeBadge.textContent = 'REJECTED — CHAMPION KEPT';
      }
    }

    const cPr  = lastEval.champion_metrics?.pr_auc || 0;
    const chPr = lastEval.challenger_metrics?.pr_auc || 0;
    const cRoc = lastEval.champion_metrics?.roc_auc || 0;
    const chRoc= lastEval.challenger_metrics?.roc_auc || 0;

    if (centerChampPr)  centerChampPr.textContent  = cPr.toFixed(4);
    if (centerChalPr)   centerChalPr.textContent   = chPr.toFixed(4);
    const dPr = chPr - cPr;
    if (centerDeltaPr) {
      centerDeltaPr.textContent = (dPr >= 0 ? '+' : '') + dPr.toFixed(4);
      centerDeltaPr.style.color = dPr >= 0 ? 'var(--text-approved)' : 'var(--text-halt)';
    }

    if (centerChampRoc) centerChampRoc.textContent = cRoc.toFixed(4);
    if (centerChalRoc)  centerChalRoc.textContent  = chRoc.toFixed(4);
    const dRoc = chRoc - cRoc;
    if (centerDeltaRoc) {
      centerDeltaRoc.textContent = (dRoc >= 0 ? '+' : '') + dRoc.toFixed(4);
      centerDeltaRoc.style.color = dRoc >= 0 ? 'var(--text-approved)' : 'var(--text-halt)';
    }

    if (centerOutcomeReason) {
      centerOutcomeReason.textContent = lastEval.promoted
        ? `Validated & Promoted: Exceeded baseline PR-AUC standards (${chPr.toFixed(4)}) on validation split.`
        : `Champion Preserved: ${lastEval.reason || 'Challenger did not satisfy strict production criteria.'}`;
    }
  }

  // 3. Model Registry History Table
  if (modelHistoryTableBody) {
    modelHistoryTableBody.innerHTML = '';
    if (history.length === 0) {
      modelHistoryTableBody.innerHTML = `
        <tr><td colspan="8" class="admin-empty-cell">
          <div class="admin-empty-state"><span>No model evaluations logged yet.</span></div>
        </td></tr>`;
      return;
    }

    // Newest first
    const reversed = [...history].reverse();
    reversed.forEach(item => {
      const isBase = !item.challenger_version;
      const vName  = isBase ? item.model_version : item.challenger_version;
      const role   = isBase ? 'Base Model' : 'Challenger';
      const ts     = fmtTs(item.evaluated_at || item.promoted_at);
      const fbUsed = item.feedback_samples !== undefined ? `${item.feedback_samples} samples` : '--';
      const pr     = item.challenger_metrics?.pr_auc !== undefined ? item.challenger_metrics.pr_auc.toFixed(4) : '--';
      const roc    = item.challenger_metrics?.roc_auc !== undefined ? item.challenger_metrics.roc_auc.toFixed(4) : '--';

      let statusBadge = '';
      if (item.promoted) {
        statusBadge = '<span class="feedback-pill legit">PROMOTED</span>';
      } else {
        statusBadge = '<span class="feedback-pill fraud">REJECTED</span>';
      }

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="font-weight:600; color:#fff;">${vName}</td>
        <td><span class="view-tab-badge">${role}</span></td>
        <td style="white-space:nowrap;">${ts}</td>
        <td>${fbUsed}</td>
        <td style="color:var(--text-approved); font-weight:600;">${pr}</td>
        <td style="color:var(--color-velocity); font-weight:600;">${roc}</td>
        <td>${statusBadge}</td>
        <td style="color:var(--text-secondary); max-width:240px; overflow:hidden; text-overflow:ellipsis;" title="${item.reason || ''}">
          ${item.reason || '--'}
        </td>
      `;
      modelHistoryTableBody.appendChild(tr);
    });
  }
}

if (btnManualTriggerRetrain) {
  btnManualTriggerRetrain.addEventListener('click', openRetrainModal);
}

// ── Merge API Data into LocalStorage (Non-destructive) ───────────────────────
function mergeApiData(apiEntries) {
  const local = getLocalStore();
  let changed = false;

  apiEntries.forEach(apiTx => {
    const idx = local.findIndex(t => t.transaction_id === apiTx.transaction_id);
    if (idx === -1) {
      local.unshift(apiTx);
      changed = true;
    } else {
      // Sync override and verified label if API has newer data
      if (apiTx.approved_by_admin && !local[idx].approved_by_admin) {
        local[idx].approved_by_admin = true;
        local[idx].approved_at       = apiTx.approved_at;
        local[idx].current_decision  = 'APPROVE';
        changed = true;
      }
      if (apiTx.verified_label !== null && local[idx].verified_label !== apiTx.verified_label) {
        local[idx].verified_label = apiTx.verified_label;
        local[idx].admin_note     = apiTx.admin_note;
        changed = true;
      }
    }
  });

  if (changed) {
    local.sort((a, b) => new Date(b.flagged_at) - new Date(a.flagged_at));
    if (local.length > 500) local.splice(500);
    saveLocalStore(local);
  }
  return local;
}

// ── Load Data ─────────────────────────────────────────────────────────────────
async function loadTransactions() {
  allTransactions = getLocalStore();
  renderTable();
  updateStats();
  updateRiskDistribution();

  try {
    const res = await fetch('/api/v1/admin/flagged?limit=200');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const apiData = await res.json();
    allTransactions = mergeApiData(apiData);
    setServerStatus(true);
  } catch (_) {
    setServerStatus(false);
  }

  renderTable();
  updateStats();
  updateRiskDistribution();
  await loadModelStatus();
  if (lastRefreshedLabel) {
    lastRefreshedLabel.textContent = 'Synced: ' + new Date().toLocaleTimeString();
  }
}

// ── Executive Stats ───────────────────────────────────────────────────────────
function updateStats() {
  const total    = allTransactions.length;
  const halt     = allTransactions.filter(t => t.original_decision === 'HALT').length;
  const review   = allTransactions.filter(t => t.original_decision === 'REVIEW').length;
  const approved = allTransactions.filter(t => t.approved_by_admin).length;
  const pending  = allTransactions.filter(t => !t.approved_by_admin).length;

  if (statTotal)    statTotal.textContent    = total;
  if (statHalt)     statHalt.textContent     = halt;
  if (statReview)   statReview.textContent   = review;
  if (statApproved) statApproved.textContent = approved;
  if (statPending)  statPending.textContent  = pending;

  // Refined subtexts with accurate ratios
  if (statTotalSub) statTotalSub.textContent = `${total} transactions in ledger`;
  if (statHaltSub) {
    const pct = total > 0 ? ((halt / total) * 100).toFixed(1) : '0.0';
    statHaltSub.textContent = `${pct}% of flagged traffic`;
  }
  if (statReviewSub) {
    const pct = total > 0 ? ((review / total) * 100).toFixed(1) : '0.0';
    statReviewSub.textContent = `${pct}% of flagged traffic`;
  }
  if (statApprovedSub) statApprovedSub.textContent = 'Human verified overrides';
  if (statPendingSub)  statPendingSub.textContent  = `${pending} require attention`;

  if (tabQueueBadge) tabQueueBadge.textContent = `${pending}`;
}

// ── Real-Time Risk Distribution ───────────────────────────────────────────────
function updateRiskDistribution() {
  const total = allTransactions.length;
  if (total === 0) {
    if (barApproveSeg) barApproveSeg.style.width = '0%';
    if (barReviewSeg)  barReviewSeg.style.width  = '0%';
    if (barHaltSeg)    barHaltSeg.style.width    = '0%';
    return;
  }

  const approved = allTransactions.filter(t => t.approved_by_admin || t.current_decision === 'APPROVE').length;
  const review   = allTransactions.filter(t => !t.approved_by_admin && t.current_decision === 'REVIEW').length;
  const halt     = allTransactions.filter(t => !t.approved_by_admin && t.current_decision === 'HALT').length;

  const pctApprove = ((approved / total) * 100).toFixed(1);
  const pctReview  = ((review / total) * 100).toFixed(1);
  const pctHalt    = ((halt / total) * 100).toFixed(1);

  if (barApproveSeg) barApproveSeg.style.width = `${pctApprove}%`;
  if (barReviewSeg)  barReviewSeg.style.width  = `${pctReview}%`;
  if (barHaltSeg)    barHaltSeg.style.width    = `${pctHalt}%`;

  if (riskApproveCount) riskApproveCount.textContent = `${approved}`;
  if (riskReviewCount)  riskReviewCount.textContent  = `${review}`;
  if (riskHaltCount)    riskHaltCount.textContent    = `${halt}`;

  if (riskApprovePct) riskApprovePct.textContent = `${pctApprove}%`;
  if (riskReviewPct)  riskReviewPct.textContent  = `${pctReview}%`;
  if (riskHaltPct)    riskHaltPct.textContent    = `${pctHalt}%`;
}

// ── Filter + Search ───────────────────────────────────────────────────────────
function getFiltered() {
  let list = allTransactions;
  if (activeFilter === 'HALT')          list = list.filter(t => t.original_decision === 'HALT');
  else if (activeFilter === 'REVIEW')   list = list.filter(t => t.original_decision === 'REVIEW');
  else if (activeFilter === 'approved') list = list.filter(t => t.approved_by_admin);
  else if (activeFilter === 'pending')  list = list.filter(t => !t.approved_by_admin);

  if (searchQuery) {
    const q = searchQuery.toLowerCase();
    list = list.filter(t => t.transaction_id.toLowerCase().includes(q));
  }
  return list;
}

// ── Render Transaction Table ──────────────────────────────────────────────────
function renderTable() {
  if (!adminTableBody) return;
  const rows = getFiltered();
  adminTableBody.innerHTML = '';

  if (rows.length === 0) {
    const msg = allTransactions.length === 0
      ? 'No flagged transactions yet. Switch to "Coordinated Attack" scenario on the live authorization feed.'
      : 'No transactions match the selected filter criteria.';
    adminTableBody.innerHTML = `
      <tr><td colspan="11" class="admin-empty-cell">
        <div class="admin-empty-state"><span>${msg}</span></div>
      </td></tr>`;
    if (tableCountLabel) tableCountLabel.textContent = '0 transactions';
    return;
  }

  if (tableCountLabel) {
    tableCountLabel.textContent = `${rows.length} transaction${rows.length !== 1 ? 's' : ''}`;
  }

  rows.forEach(tx => {
    const approved    = tx.approved_by_admin;
    const curDecision = approved ? 'APPROVE' : tx.current_decision;
    const isSelected  = tx.transaction_id === selectedTxId;

    let fbTag = '';
    if (tx.verified_label === 1) {
      fbTag = ' <span class="feedback-pill fraud" title="Verified FRAUD ground truth">FRAUD</span>';
    } else if (tx.verified_label === 0) {
      fbTag = ' <span class="feedback-pill legit" title="Verified LEGITIMATE ground truth">LEGIT</span>';
    }

    const tr = document.createElement('tr');
    tr.className    = isSelected ? 'selected-row' : '';
    tr.dataset.txid = tx.transaction_id;
    tr.style.cursor = 'pointer';

    // Semantic risk score coloring
    const rScore = parseFloat(tx.risk_score);
    let riskColor = 'var(--text-primary)';
    if (rScore >= 0.70) riskColor = 'var(--text-halt)';
    else if (rScore >= 0.35) riskColor = 'var(--text-review)';
    else riskColor = 'var(--text-approved)';

    tr.innerHTML = `
      <td><button class="row-selector-btn" data-txid="${tx.transaction_id}" title="Inspect transaction details">&#9654;</button></td>
      <td style="color:var(--text-tertiary);">${fmtTimeOnly(tx.flagged_at)}</td>
      <td style="font-family:var(--font-mono);color:var(--text-primary);font-weight:600;">
        ${tx.transaction_id}${fbTag}
      </td>
      <td style="color:var(--text-primary);font-weight:500;">${tx.amount || '--'}</td>
      <td><span class="ledger-badge ${tx.original_decision.toLowerCase()}">${tx.original_decision}</span></td>
      <td><span class="ledger-badge ${curDecision.toLowerCase()}">${curDecision}</span></td>
      <td>${parseFloat(tx.ml_fraud_prob).toFixed(4)}</td>
      <td>${parseFloat(tx.anomaly_score).toFixed(4)}</td>
      <td style="font-weight:600;color:${riskColor};">${rScore.toFixed(4)}</td>
      <td>${parseFloat(tx.velocity_surge_index).toFixed(4)}</td>
      <td>${approved
        ? '<span class="btn-row-approved">&#10003; Approved</span>'
        : `<button class="btn-row-approve" data-txid="${tx.transaction_id}">Approve</button>`
      }</td>`;

    adminTableBody.appendChild(tr);
  });

  // Event bindings
  adminTableBody.querySelectorAll('tr[data-txid]').forEach(row => {
    row.addEventListener('click', e => {
      if (e.target.tagName === 'BUTTON') return;
      openDetail(row.dataset.txid);
    });
  });
  adminTableBody.querySelectorAll('.row-selector-btn').forEach(btn => {
    btn.addEventListener('click', () => openDetail(btn.dataset.txid));
  });
  adminTableBody.querySelectorAll('.btn-row-approve').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      doApprove(btn.dataset.txid, () => {
        allTransactions = getLocalStore();
        renderTable();
        updateStats();
        updateRiskDistribution();
        if (selectedTxId === btn.dataset.txid) openDetail(btn.dataset.txid);
      });
    });
  });
}

// ── Detail Drawer Inspection ──────────────────────────────────────────────────
function openDetail(txId) {
  const tx = allTransactions.find(t => t.transaction_id === txId);
  if (!tx || !detailPanel) return;
  selectedTxId = txId;
  clearDetailMsg();

  if (adminTableBody) {
    adminTableBody.querySelectorAll('tr').forEach(r => r.classList.remove('selected-row'));
    const row = adminTableBody.querySelector(`tr[data-txid="${txId}"]`);
    if (row) row.classList.add('selected-row');
  }

  const dtId = document.getElementById('detailTxId');
  if (dtId) dtId.textContent = tx.transaction_id;

  const cur   = tx.approved_by_admin ? 'approve' : tx.current_decision.toLowerCase();
  const label = tx.approved_by_admin ? 'APPROVED' : tx.current_decision;
  const badge = document.getElementById('detailCurrentBadge');
  if (badge) {
    badge.textContent = label;
    badge.className   = 'decision-badge ' + cur;
  }

  const origDec = document.getElementById('detailOriginalDecision');
  if (origDec) origDec.textContent = tx.original_decision;

  const amt = document.getElementById('detailAmount');
  if (amt) amt.textContent = tx.amount || '--';

  const flagged = document.getElementById('detailFlaggedAt');
  if (flagged) flagged.textContent = fmtTs(tx.flagged_at);

  const ml = document.getElementById('detailML');
  if (ml) ml.textContent = parseFloat(tx.ml_fraud_prob).toFixed(4);

  const anom = document.getElementById('detailAnomaly');
  if (anom) anom.textContent = parseFloat(tx.anomaly_score).toFixed(4);

  const rk = document.getElementById('detailRisk');
  if (rk) rk.textContent = parseFloat(tx.risk_score).toFixed(4);

  const vel = document.getElementById('detailVelocity');
  if (vel) vel.textContent = parseFloat(tx.velocity_surge_index).toFixed(4);

  const approvedEl = document.getElementById('detailApprovedStatus');
  if (approvedEl) {
    if (tx.approved_by_admin) {
      approvedEl.textContent = 'Yes — ' + fmtTs(tx.approved_at);
      approvedEl.style.color = 'var(--text-approved)';
    } else {
      approvedEl.textContent = 'No';
      approvedEl.style.color = '';
    }
  }

  const list = document.getElementById('detailReasonsList');
  if (list) {
    list.innerHTML = '';
    (tx.reasons || []).forEach(r => {
      const li = document.createElement('li');
      li.textContent = r;
      list.appendChild(li);
    });
  }

  // Operational Override Button
  if (detailApproveBtn && detailApproveBtnLabel) {
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
        updateRiskDistribution();
        openDetail(txId);
      });
    }
  }
  if (detailApproveStatus) detailApproveStatus.textContent = '';

  // Setup Human-in-the-Loop Feedback Controls
  selectedFeedbackLabel = (tx.verified_label !== undefined && tx.verified_label !== null) ? tx.verified_label : null;
  if (feedbackNoteInput) feedbackNoteInput.value = tx.admin_note || '';
  if (feedbackSubmitMsg) feedbackSubmitMsg.textContent = '';

  updateFeedbackFormUI();

  detailPanel.style.display = 'block';
  detailPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function updateFeedbackFormUI() {
  if (btnLabelLegit) btnLabelLegit.classList.toggle('selected', selectedFeedbackLabel === 0);
  if (btnLabelFraud) btnLabelFraud.classList.toggle('selected', selectedFeedbackLabel === 1);

  if (selectedFeedbackLabel !== null) {
    if (btnSubmitFeedback) btnSubmitFeedback.disabled = false;
    if (feedbackStatusBadge) {
      feedbackStatusBadge.style.display = 'inline-block';
      if (selectedFeedbackLabel === 1) {
        feedbackStatusBadge.className = 'feedback-pill fraud';
        feedbackStatusBadge.textContent = '✓ FRAUD GROUND TRUTH';
      } else {
        feedbackStatusBadge.className = 'feedback-pill legit';
        feedbackStatusBadge.textContent = '✓ LEGITIMATE GROUND TRUTH';
      }
    }
    if (btnSubmitFeedbackLabel) btnSubmitFeedbackLabel.textContent = 'Update Verified Feedback';
  } else {
    if (btnSubmitFeedback) btnSubmitFeedback.disabled = true;
    if (feedbackStatusBadge) feedbackStatusBadge.style.display = 'none';
    if (btnSubmitFeedbackLabel) btnSubmitFeedbackLabel.textContent = 'Submit Verified Feedback';
  }
}

function closeDetail() {
  if (detailPanel) detailPanel.style.display = 'none';
  selectedTxId = null;
  if (adminTableBody) {
    adminTableBody.querySelectorAll('tr').forEach(r => r.classList.remove('selected-row'));
  }
}

// ── Submit Verified Feedback ──────────────────────────────────────────────────
async function submitVerifiedFeedback() {
  if (selectedFeedbackLabel === null || !selectedTxId) return;

  if (btnSubmitFeedback) btnSubmitFeedback.disabled = true;
  if (btnSubmitFeedbackLabel) btnSubmitFeedbackLabel.textContent = 'Recording to SQLite...';
  if (feedbackSubmitMsg) feedbackSubmitMsg.textContent = '';

  const note = feedbackNoteInput ? feedbackNoteInput.value.trim() : '';

  try {
    const res = await fetch(`/api/v1/admin/feedback/${encodeURIComponent(selectedTxId)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        verified_label: selectedFeedbackLabel,
        admin_note: note,
        allow_override: true,
        reviewer_id: 'analyst_console'
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Feedback submission failed');
    }

    const data = await res.json();

    // Update in local store
    const store = getLocalStore();
    const idx = store.findIndex(t => t.transaction_id === selectedTxId);
    if (idx !== -1) {
      store[idx].verified_label = selectedFeedbackLabel;
      store[idx].admin_note     = note;
      saveLocalStore(store);
      allTransactions = store;
    }

    if (feedbackSubmitMsg) {
      feedbackSubmitMsg.textContent = `✓ Recorded as ${data.label_name}! Total training pool: ${data.total_feedback_count}/${data.retrain_threshold}`;
      feedbackSubmitMsg.style.color = 'var(--text-approved)';
    }
    if (btnSubmitFeedbackLabel) btnSubmitFeedbackLabel.textContent = 'Update Verified Feedback';
    if (btnSubmitFeedback) btnSubmitFeedback.disabled = false;

    updateFeedbackFormUI();
    renderTable();
    await loadModelStatus();

  } catch (err) {
    if (feedbackSubmitMsg) {
      feedbackSubmitMsg.textContent = `Error: ${err.message}`;
      feedbackSubmitMsg.style.color = 'var(--text-halt)';
    }
    if (btnSubmitFeedback) btnSubmitFeedback.disabled = false;
    if (btnSubmitFeedbackLabel) btnSubmitFeedbackLabel.textContent = 'Retry Submission';
  }
}

// ── Operational Override (Approve) ───────────────────────────────────────────
function doApprove(txId, onDone) {
  const store = getLocalStore();
  const idx   = store.findIndex(t => t.transaction_id === txId);
  if (idx !== -1 && !store[idx].approved_by_admin) {
    const now = new Date().toISOString();
    store[idx].approved_by_admin = true;
    store[idx].approved_at       = now;
    store[idx].current_decision  = 'APPROVE';
    saveLocalStore(store);
  }

  if (detailApproveBtn) detailApproveBtn.disabled = true;
  if (detailApproveBtnLabel) detailApproveBtnLabel.textContent = 'Approved!';
  if (detailApproveStatus) detailApproveStatus.textContent = 'Override recorded at ' + new Date().toLocaleTimeString();
  showDetailMsg('Transaction approved operationally. Notice: ML ground truth remains separate.', 'info');

  if (onDone) onDone();

  fetch('/api/v1/admin/approve/' + encodeURIComponent(txId), { method: 'POST' })
    .catch(() => {});
}

// ── Retraining Modal Flow ─────────────────────────────────────────────────────
function openRetrainModal() {
  if (!retrainModal) return;
  retrainModal.style.display = 'flex';
  if (retrainLoadingState) retrainLoadingState.style.display = 'none';
  if (retrainResultsState) retrainResultsState.style.display = 'none';
  if (modalActionBtn) {
    modalActionBtn.textContent = 'Execute Retraining Cycle';
    modalActionBtn.disabled = false;
  }
}

function closeRetrainModal() {
  if (retrainModal) retrainModal.style.display = 'none';
}

async function executeRetrainingCycle() {
  if (!modalActionBtn) return;
  if (modalActionBtn.textContent === 'Close') {
    closeRetrainModal();
    return;
  }

  modalActionBtn.disabled = true;
  if (retrainLoadingState) retrainLoadingState.style.display = 'block';
  if (retrainResultsState) retrainResultsState.style.display = 'none';
  if (retrainProgressText) retrainProgressText.textContent = 'Training Challenger Model with LightGBM...';

  try {
    const res = await fetch('/api/v1/admin/retrain', { method: 'POST' });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Retraining failed');
    }
    const result = await res.json();

    if (retrainLoadingState) retrainLoadingState.style.display = 'none';
    if (retrainResultsState) retrainResultsState.style.display = 'block';

    if (modalChampionName)   modalChampionName.textContent   = result.previous_model || 'model_v2';
    if (modalChallengerName) modalChallengerName.textContent = result.candidate_model || 'model_v3';
    if (modalFeedbackUsed)   modalFeedbackUsed.textContent   = `${result.feedback_samples} samples`;

    if (modalPromotionBadge) {
      if (result.promoted) {
        modalPromotionBadge.className = 'feedback-pill legit';
        modalPromotionBadge.textContent = 'PROMOTED TO ACTIVE';
      } else {
        modalPromotionBadge.className = 'feedback-pill fraud';
        modalPromotionBadge.textContent = 'REJECTED (CHAMPION KEPT)';
      }
    }

    const champPr  = result.champion_metrics ? result.champion_metrics.pr_auc : 0;
    const chalPr   = result.challenger_metrics ? result.challenger_metrics.pr_auc : 0;
    const champRoc = result.champion_metrics ? result.champion_metrics.roc_auc : 0;
    const chalRoc  = result.challenger_metrics ? result.challenger_metrics.roc_auc : 0;

    if (rowChampPr) rowChampPr.textContent = champPr.toFixed(4);
    if (rowChalPr)  rowChalPr.textContent  = chalPr.toFixed(4);
    const deltaPr = (chalPr - champPr);
    if (rowDeltaPr) {
      rowDeltaPr.textContent = (deltaPr >= 0 ? '+' : '') + deltaPr.toFixed(4);
      rowDeltaPr.style.color = deltaPr >= 0 ? 'var(--text-approved)' : 'var(--text-halt)';
    }

    if (rowChampRoc) rowChampRoc.textContent = champRoc.toFixed(4);
    if (rowChalRoc)  rowChalRoc.textContent  = chalRoc.toFixed(4);
    const deltaRoc = (chalRoc - champRoc);
    if (rowDeltaRoc) {
      rowDeltaRoc.textContent = (deltaRoc >= 0 ? '+' : '') + deltaRoc.toFixed(4);
      rowDeltaRoc.style.color = deltaRoc >= 0 ? 'var(--text-approved)' : 'var(--text-halt)';
    }

    if (modalReasonText) {
      modalReasonText.textContent = result.promoted
        ? `Validation Passed: Challenger exceeded production benchmark standards (${chalPr.toFixed(4)}) and was atomically promoted to Champion.`
        : `Champion Preserved: ${result.reason || 'Challenger did not satisfy promotion criteria.'}`;
    }

    modalActionBtn.textContent = 'Close';
    modalActionBtn.disabled = false;

    // Refresh model status across entire console
    await loadModelStatus();

  } catch (err) {
    if (retrainLoadingState) retrainLoadingState.style.display = 'none';
    if (retrainResultsState) retrainResultsState.style.display = 'block';
    if (modalPromotionBadge) {
      modalPromotionBadge.className = 'feedback-pill fraud';
      modalPromotionBadge.textContent = 'FAILED';
    }
    if (modalReasonText) modalReasonText.textContent = `Retraining error: ${err.message}`;
    modalActionBtn.textContent = 'Close';
    modalActionBtn.disabled = false;
  }
}

// ── Auto-Refresh ──────────────────────────────────────────────────────────────
function startAutoRefresh() {
  if (autoRefreshTimer) clearInterval(autoRefreshTimer);
  autoRefreshTimer = setInterval(loadTransactions, AUTO_REFRESH_MS);
}

// ── Event Wiring ──────────────────────────────────────────────────────────────
if (searchInput) {
  searchInput.addEventListener('input', () => {
    searchQuery = searchInput.value.trim();
    renderTable();
  });
}

document.querySelectorAll('.filter-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    activeFilter = tab.dataset.filter;
    renderTable();
  });
});

if (refreshBtn) refreshBtn.addEventListener('click', loadTransactions);
if (detailCloseBtn) detailCloseBtn.addEventListener('click', closeDetail);

// Feedback Label Choices
if (btnLabelLegit) {
  btnLabelLegit.addEventListener('click', () => {
    selectedFeedbackLabel = 0;
    updateFeedbackFormUI();
  });
}
if (btnLabelFraud) {
  btnLabelFraud.addEventListener('click', () => {
    selectedFeedbackLabel = 1;
    updateFeedbackFormUI();
  });
}
if (btnSubmitFeedback) btnSubmitFeedback.addEventListener('click', submitVerifiedFeedback);

// Retraining Modal Events
if (btnRetrainModal) btnRetrainModal.addEventListener('click', openRetrainModal);
if (closeRetrainModalBtn) closeRetrainModalBtn.addEventListener('click', closeRetrainModal);
if (modalActionBtn) modalActionBtn.addEventListener('click', executeRetrainingCycle);

// ── Boot ──────────────────────────────────────────────────────────────────────
loadTransactions();
startAutoRefresh();
