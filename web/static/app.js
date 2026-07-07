/* ── WebSocket ─────────────────────────────────────────────── */
let ws = null;
let awaitingInput = false;   // true when agent is blocked on get_input()
let _awaitingInterrupted = false;  // user sent a message while awaitingInput was true
let _lastAgentMsg = null;          // {text, isBrain} — last AI bubble before awaiting_input

/* ── Command history (CLI-style up/down navigation) ────────── */
const _cmdHistory = [];
let _histIdx = -1;       // -1 = not browsing history
let _histDraft = '';     // saves current draft when user starts browsing
let activeAgent = null;
let _pendingAdhocClear = false;   // cleared on next popup open after user sends a message

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => setStatus('연결됨', '');

  ws.onmessage = (evt) => {
    _msgQueue.push(JSON.parse(evt.data));
    if (!_rafPending) {
      _rafPending = true;
      requestAnimationFrame(_flushQueue);
    }
  };

  ws.onclose = () => {
    setStatus('연결 끊김', 'error');
    setTimeout(connectWS, 2000);   // auto-reconnect
  };

  ws.onerror = () => ws.close();
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN)
    ws.send(JSON.stringify(obj));
}

/* ── rAF message queue ─────────────────────────────────────── */
let _msgQueue   = [];
let _rafPending = false;
let _scrollTargets = new Set();

function _flushQueue() {
  _rafPending = false;
  const msgs = _msgQueue.splice(0);
  for (const msg of msgs) handleMessage(msg);
  _scrollTargets.forEach(el => { el.scrollTop = el.scrollHeight; });
  _scrollTargets.clear();
}

/* ── Message handling ──────────────────────────────────────── */
function handleMessage(msg) {
  const isBrain = msg.source === 'brain';

  switch (msg.type) {

    case 'ai_message':
      appendAI(msg.content, isBrain);
      break;

    case 'tool_output':
      if (isBrain) {
        openAdhoc();
        adhocAppendToolOutput(msg.content);
      } else {
        appendToolOutput(msg.content);
      }
      break;

    case 'plot':
      if (isBrain) {
        adhocAppendPlot(msg.filename);   // adhocAppendPlot calls openAdhoc internally
      } else {
        appendPlot(msg.filename);
      }
      break;

    case 'html_content':
      if (isBrain) {
        openAdhoc();
        adhocAppendHtml(msg.content);
      } else {
        appendHtml(msg.content);
      }
      break;

    case 'dqm_canvases':
      if (isBrain) {
        openAdhoc();
        adhocDrawDqmCanvases(msg.base_prefix, msg.canvases);
      }
      break;

    case 'adhoc_clarify':
      showClarifyInChat(msg.question || '');
      break;

    case 'adhoc_confirm':
      adhocShowConfirm(msg.preview || '', msg.tool || '', !!msg.scenario_running);
      break;

    case 'awaiting_input':
      awaitingInput = true;
      _lastAgentMsg = _pendingAgentMsg;
      addCompleteButton();
      break;

    case 'awaiting_hv_confirm':
      awaitingInput = true;
      _lastAgentMsg = _pendingAgentMsg;
      addHvConfirmButtons();
      break;

    case 'status':
      if (!isBrain) {
        setStatus(msg.content, msg.content.includes('완료') ? '' : 'running');
      }
      break;

    case 'daq_complete':
      console.log('[autoTB] daq_complete received');
      playDaqCompleteSound();
      break;

    case 'agent_done':
      awaitingInput = false;
      removeCompleteButtons();
      document.querySelectorAll('.inline-retry-btn').forEach(btn => {
        btn.disabled = true;
        btn.textContent = '종료됨';
      });
      setAgentButtons(false);
      activeAgent = null;
      setStatus('대기 중', '');
      break;

    case 'open_window':
      adhocAppendLink(msg.url, msg.label || msg.url);
      break;

    case 'dqm_live_start':
      dqmLiveStart(msg);
      break;

    case 'dqm_refresh':
      dqmRefresh(msg);
      break;

    case 'dqm_live_end':
      dqmLiveEnd(msg);
      break;

    case 'error':
      if (isBrain) {
        adhocAppendToolOutput('❌ ' + msg.content, true);
      } else {
        appendToolOutput('❌ ' + msg.content, true);
      }
      break;

    case 'tool_error':
      appendToolError(msg.tool_name, msg.error, msg.attempts, isBrain);
      break;
  }
}

/* ── User actions ──────────────────────────────────────────── */
function addCompleteButton() {
  removeCompleteButtons();   // only one at a time
  const row = document.createElement('div');
  row.className = 'complete-row';

  if (_awaitingInterrupted && _lastAgentMsg) {
    _awaitingInterrupted = false;
    appendAI(_lastAgentMsg.text, _lastAgentMsg.isBrain);
  } else {
    _awaitingInterrupted = false;
  }

  const btn = document.createElement('button');
  btn.className = 'inline-complete-btn';
  btn.textContent = '✔ 완료';
  btn.onclick = sendComplete;
  row.appendChild(btn);
  chatScroll().appendChild(row);
  scrollBottom(chatScroll());
}

function removeCompleteButtons() {
  chatScroll().querySelectorAll('.complete-row').forEach(el => el.remove());
}

function addHvConfirmButtons() {
  removeCompleteButtons();
  const row = document.createElement('div');
  row.className = 'complete-row hv-confirm-row';

  if (_awaitingInterrupted && _lastAgentMsg) {
    _awaitingInterrupted = false;
    appendAI(_lastAgentMsg.text, _lastAgentMsg.isBrain);
  } else {
    _awaitingInterrupted = false;
  }

  const doneBtn = document.createElement('button');
  doneBtn.className = 'inline-complete-btn';
  doneBtn.textContent = '✔ 완료';
  doneBtn.onclick = sendComplete;

  const modBtn = document.createElement('button');
  modBtn.className = 'inline-modify-btn';
  modBtn.textContent = '✏ 수정';

  const modArea = document.createElement('div');
  modArea.className = 'hv-modify-area';
  modArea.style.display = 'none';

  const modInput = document.createElement('input');
  modInput.type = 'text';
  modInput.className = 'hv-modify-input';
  modInput.placeholder = '예) C 30 올려  /  모두 40 올려  /  C=790';

  const sendBtn = document.createElement('button');
  sendBtn.className = 'inline-complete-btn';
  sendBtn.textContent = '전송';
  sendBtn.onclick = () => {
    const text = modInput.value.trim();
    if (!text) return;
    removeCompleteButtons();
    awaitingInput = false;
    appendUserBubble(text);
    send({ type: 'user_input', content: text });
  };

  modInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') sendBtn.click();
  });

  modBtn.onclick = () => {
    modArea.style.display = modArea.style.display === 'none' ? 'flex' : 'none';
    if (modArea.style.display === 'flex') modInput.focus();
  };

  modArea.appendChild(modInput);
  modArea.appendChild(sendBtn);
  row.appendChild(doneBtn);
  row.appendChild(modBtn);
  row.appendChild(modArea);
  chatScroll().appendChild(row);
  scrollBottom(chatScroll());
}

function sendComplete() {
  removeCompleteButtons();
  awaitingInput = false;
  appendUserBubble('완료');
  send({ type: 'user_input', content: '완료' });
}

function sendText() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text) return;
  // Save to history (avoid duplicate consecutive entries)
  if (_cmdHistory[0] !== text) _cmdHistory.unshift(text);
  _histIdx = -1;
  _histDraft = '';
  input.value = '';
  _pendingAdhocClear = true;   // next popup open will clear previous results
  appendUserBubble(text);
  send({ type: 'user_input', content: text });
  // If agent was waiting, remove the inline 완료 button too
  if (awaitingInput) _awaitingInterrupted = true;
  removeCompleteButtons();
  awaitingInput = false;
}

function startAgent(agentName) {
  if (activeAgent) {
    alert('에이전트가 이미 실행 중입니다. 먼저 Stop을 클릭하세요.');
    return;
  }
  const needsTowerPick = ['em_scan', 'calib_scan', 'hv_equalization', 'hv_equalization_sim', 'position_scan', 'position_scan_sim'];
  if (needsTowerPick.includes(agentName)) {
    openTowerPicker(agentName);
    return;
  }
  _launchAgent(agentName, {});
}

function _launchAgent(agentName, params) {
  clearPanels();
  activeAgent = agentName;
  setAgentButtons(true);
  setStatus(`${agentName} 에이전트 실행 중`, 'running');
  send({ type: 'start_agent', agent: agentName, params });
}

function stopAgent() {
  send({ type: 'stop_agent' });
}

function killRun() {
  const btn = document.getElementById('kill-btn');
  btn.disabled = true;
  send({ type: 'kill_run' });
  setTimeout(() => { btn.disabled = false; }, 2000);
}

function openHvCheck() {
  window.open('/hv/check', '_blank', 'width=1100,height=820');
}

/* ── DOM limits ────────────────────────────────────────────── */
const MAX_BLOCK_LINES  = 400;   // tool-block 한 개 내 최대 줄 수
const MAX_TOOL_BLOCKS  = 80;    // right-scroll 최대 블록 수
const MAX_CHAT_NODES   = 120;   // chat-scroll 최대 노드 수
const MAX_ADHOC_NODES  = 80;    // adhoc-scroll 최대 노드 수

function trimContainer(el, max) {
  while (el.childElementCount > max)
    el.removeChild(el.firstElementChild);
}

function trimBlockLines(block) {
  const lines = block.textContent.split('\n');
  if (lines.length > MAX_BLOCK_LINES)
    block.textContent = lines.slice(-Math.floor(MAX_BLOCK_LINES / 2)).join('\n');
}

/* ── Clear helpers ─────────────────────────────────────────── */
function clearToolOutput() { rightScroll().innerHTML = ''; }
function clearChat()       { chatScroll().innerHTML = ''; }
function clearAdhoc()      { adhocScroll().innerHTML = ''; }

/* ── DOM helpers ───────────────────────────────────────────── */
let _pendingAgentMsg = null;  // last AI message seen, candidate for re-display

function appendAI(text, isBrain = false) {
  _pendingAgentMsg = { text, isBrain };
  const div = document.createElement('div');
  div.className = isBrain ? 'ai-bubble brain-bubble' : 'ai-bubble';
  if (isBrain) {
    const tag = document.createElement('span');
    tag.className = 'brain-tag';
    tag.textContent = 'Background';
    div.appendChild(tag);
    div.appendChild(document.createTextNode(' ' + text));
  } else {
    div.textContent = text;
  }
  chatScroll().appendChild(div);
  trimContainer(chatScroll(), MAX_CHAT_NODES);
  scrollBottom(chatScroll());
}

/* ── Inline clarify (AI message form in chat) ─────────────── */
function showClarifyInChat(question) {
  const div = document.createElement('div');
  div.className = 'ai-bubble brain-bubble clarify-inline';

  const qLine = document.createElement('div');
  const tag = document.createElement('span');
  tag.className = 'brain-tag';
  tag.textContent = '추가 정보 필요';
  qLine.appendChild(tag);
  qLine.appendChild(document.createTextNode(' ' + question));
  div.appendChild(qLine);

  const row = document.createElement('div');
  row.className = 'clarify-inline-row';

  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'clarify-inline-input';
  input.placeholder = '답변을 입력하세요...';

  const btn = document.createElement('button');
  btn.className = 'clarify-inline-send';
  btn.textContent = '전송';

  const submit = () => {
    const text = input.value.trim();
    if (!text) return;
    input.disabled = true;
    btn.disabled = true;
    appendUserBubble(text);
    send({ type: 'user_input', content: text });
  };

  btn.onclick = submit;
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); submit(); }
  });

  row.appendChild(input);
  row.appendChild(btn);
  div.appendChild(row);

  chatScroll().appendChild(div);
  trimContainer(chatScroll(), MAX_CHAT_NODES);
  scrollBottom(chatScroll());
  setTimeout(() => input.focus(), 50);
}

function appendUserBubble(text) {
  const div = document.createElement('div');
  div.className = 'user-bubble';
  div.textContent = text;
  chatScroll().appendChild(div);
  trimContainer(chatScroll(), MAX_CHAT_NODES);
  scrollBottom(chatScroll());
}

function appendToolError(toolName, errorMsg, attempts, isBrain = false) {
  const div = document.createElement('div');
  div.className = 'tool-error-bubble';
  const title = document.createElement('span');
  title.className = 'tool-error-title';
  title.textContent = `Tool 오류: ${toolName}`;
  div.appendChild(title);
  div.appendChild(document.createTextNode(`${attempts}회 시도 모두 실패\n${errorMsg}`));
  chatScroll().appendChild(div);

  const retryRow = document.createElement('div');
  retryRow.className = 'retry-row';

  const retryBtn = document.createElement('button');
  retryBtn.className = 'inline-retry-btn';
  retryBtn.textContent = '↻ 다시 시도';

  const skipBtn = isBrain ? null : document.createElement('button');
  if (skipBtn) {
    skipBtn.className = 'inline-retry-btn inline-skip-btn';
    skipBtn.textContent = '→ 다음 단계로';
  }

  const disableBoth = () => {
    retryBtn.disabled = true;
    if (skipBtn) skipBtn.disabled = true;
  };
  retryBtn.onclick = () => {
    disableBoth();
    retryBtn.textContent = '재시도 중...';
    send({ type: 'user_input', content: 'retry' });
  };
  if (skipBtn) {
    skipBtn.onclick = () => {
      disableBoth();
      skipBtn.textContent = '건너뜀...';
      send({ type: 'user_input', content: 'skip' });
    };
  }

  retryRow.appendChild(retryBtn);
  if (skipBtn) retryRow.appendChild(skipBtn);
  chatScroll().appendChild(retryRow);
  trimContainer(chatScroll(), MAX_CHAT_NODES);
  scrollBottom(chatScroll());
}

function stripAnsi(text) {
  return text.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '');
}

function appendToolOutput(text, isError = false) {
  const right = rightScroll();
  const last = right.lastElementChild;
  const clean = stripAnsi(text);
  if (last && last.classList.contains('tool-block') && !last.classList.contains('error') && !isError) {
    last.textContent += '\n' + clean;
    trimBlockLines(last);
  } else {
    const div = document.createElement('div');
    div.className = 'tool-block' + (isError ? ' error' : '');
    div.textContent = clean;
    right.appendChild(div);
    trimContainer(right, MAX_TOOL_BLOCKS);
  }
  scrollBottom(right);
}

function appendPlot(filename) {
  const card = document.createElement('div');
  card.className = 'plot-card';

  const img = document.createElement('img');
  // Add timestamp to bust cache if re-generated
  img.src = `/plots/${filename}?t=${Date.now()}`;
  img.alt = filename;
  img.onclick = () => openLightbox(img.src);

  const label = document.createElement('div');
  label.className = 'plot-label';
  label.textContent = filename;

  card.appendChild(img);
  card.appendChild(label);
  rightScroll().appendChild(card);
  scrollBottom(rightScroll());
}

function clearPanels() {
  chatScroll().innerHTML = '';
  rightScroll().innerHTML = '';
}

function setStatus(text, cls) {
  const pill = document.getElementById('status-pill');
  pill.textContent = text;
  pill.className = cls ? `${cls}` : '';
}

function setAgentButtons(running) {
  ['btn-em', 'btn-calib', 'btn-hv', 'btn-hv-sim', 'btn-pos', 'btn-pos-sim'].forEach(id => {
    const btn = document.getElementById(id);
    btn.disabled = running;
    btn.classList.toggle('active', running && id === 'btn-' + agentIdOf(activeAgent));
  });
  document.getElementById('stop-btn').style.display = running ? 'inline-block' : 'none';
}

function agentIdOf(name) {
  return { em_scan: 'em', calib_scan: 'calib', hv_equalization: 'hv', hv_equalization_sim: 'hv-sim', position_scan: 'pos', position_scan_sim: 'pos-sim' }[name] || '';
}

function chatScroll()  { return document.getElementById('chat-scroll'); }
function rightScroll() { return document.getElementById('right-scroll'); }
function scrollBottom(el) { _scrollTargets.add(el); }

/* ── Ad-hoc result popup ─────────────────────────────────────── */
function adhocScroll() { return document.getElementById('adhoc-scroll'); }

function openAdhoc(clearContent = false) {
  const overlay = document.getElementById('adhoc-overlay');
  if (_pendingAdhocClear) {
    adhocScroll().innerHTML = '';
    _pendingAdhocClear = false;
  }
  if (!overlay.classList.contains('open')) {
    if (clearContent) adhocScroll().innerHTML = '';
    overlay.classList.add('open');
  }
}

function closeAdhoc() {
  document.getElementById('adhoc-overlay').classList.remove('open');
}

/* ── Clarify popup ─────────────────────────────────────────── */
function showClarifyPopup(question) {
  document.getElementById('clarify-question').textContent = question;
  document.getElementById('clarify-input').value = '';
  document.getElementById('clarify-overlay').classList.add('open');
  setTimeout(() => document.getElementById('clarify-input').focus(), 50);
}

function closeClarify() {
  document.getElementById('clarify-overlay').classList.remove('open');
}

function sendClarify() {
  const input = document.getElementById('clarify-input');
  const text = input.value.trim();
  if (!text) return;
  closeClarify();
  appendUserBubble(text);
  send({ type: 'user_input', content: text });
}

document.addEventListener('DOMContentLoaded', () => {
  const clarifyInput = document.getElementById('clarify-input');
  if (clarifyInput) {
    clarifyInput.addEventListener('keydown', e => {
      if (e.key === 'Enter') sendClarify();
      if (e.key === 'Escape') closeClarify();
    });
  }
});

function adhocAppendToolOutput(text, isError = false) {
  openAdhoc();
  const scroll = adhocScroll();
  const last = scroll.lastElementChild;
  const clean = stripAnsi(text);
  if (last && last.classList.contains('adhoc-tool-block') && !last.classList.contains('error') && !isError) {
    last.textContent += '\n' + clean;
    trimBlockLines(last);
  } else {
    const div = document.createElement('div');
    div.className = 'adhoc-tool-block' + (isError ? ' error' : '');
    div.textContent = clean;
    scroll.appendChild(div);
    trimContainer(scroll, MAX_ADHOC_NODES);
  }
  scroll.scrollTop = scroll.scrollHeight;
}

function adhocAppendHtml(html) {
  openAdhoc();
  const scroll = adhocScroll();
  const div = document.createElement('div');
  div.className = 'adhoc-tool-block';
  div.innerHTML = html;
  scroll.appendChild(div);
  scroll.scrollTop = scroll.scrollHeight;
}

function appendHtml(html) {
  const right = rightScroll();
  const div = document.createElement('div');
  div.className = 'tool-block';
  div.innerHTML = html;
  right.appendChild(div);
  scrollBottom(right);
}

function adhocShowConfirm(preview, _tool, scenarioRunning = false) {
  openAdhoc();
  const scroll = adhocScroll();

  // Remove any existing confirm card (only one at a time)
  scroll.querySelectorAll('.adhoc-confirm-card').forEach(el => el.remove());

  const card = document.createElement('div');
  card.className = 'adhoc-confirm-card';

  const title = document.createElement('div');
  title.className = 'adhoc-confirm-title';
  title.textContent = '⚠️ 확인이 필요합니다';
  card.appendChild(title);

  const body = document.createElement('div');
  body.className = 'adhoc-confirm-body';
  body.textContent = preview;
  card.appendChild(body);

  const btnRow = document.createElement('div');
  btnRow.className = 'adhoc-confirm-btns';

  const yesBtn = document.createElement('button');
  yesBtn.className = 'adhoc-confirm-yes';
  yesBtn.textContent = '✔ 확인';
  yesBtn.onclick = () => {
    send({ type: 'adhoc_confirm', confirmed: true });
    card.remove();
    if (!scenarioRunning) closeAdhoc();
  };

  const noBtn = document.createElement('button');
  noBtn.className = 'adhoc-confirm-no';
  noBtn.textContent = '✕ 취소';
  noBtn.onclick = () => {
    send({ type: 'adhoc_confirm', confirmed: false });
    card.remove();
    if (!scenarioRunning) closeAdhoc();
  };

  btnRow.appendChild(yesBtn);
  btnRow.appendChild(noBtn);
  card.appendChild(btnRow);

  scroll.appendChild(card);
  scroll.scrollTop = scroll.scrollHeight;
}

function adhocAppendLink(url, label) {
  openAdhoc();
  const scroll = adhocScroll();

  const wrap = document.createElement('div');
  wrap.className = 'adhoc-tool-block';

  const a = document.createElement('a');
  a.href = url;
  a.target = '_blank';
  a.rel = 'noopener';
  a.textContent = '🔗 ' + label;
  a.style.cssText = 'color:#4f8ef7;text-decoration:underline;cursor:pointer;font-size:13px;';

  wrap.appendChild(a);
  scroll.appendChild(wrap);
  scroll.scrollTop = scroll.scrollHeight;
}

function adhocAppendPlot(filename) {
  openAdhoc();
  const scroll = adhocScroll();

  const card = document.createElement('div');
  card.className = 'adhoc-plot-card';

  const img = document.createElement('img');
  img.src = `/plots/${filename}?t=${Date.now()}`;
  img.alt = filename;
  img.onclick = () => openLightbox(img.src);

  const label = document.createElement('div');
  label.className = 'plot-label';
  label.textContent = filename;

  card.appendChild(img);
  card.appendChild(label);
  scroll.appendChild(card);
  scroll.scrollTop = scroll.scrollHeight;
}

function adhocDrawDqmCanvases(base_prefix, canvases) {
  openAdhoc();
  const scroll = adhocScroll();

  // Label header
  const header = document.createElement('div');
  header.className = 'adhoc-tool-block';
  header.textContent = `DQM: ${base_prefix} (${canvases.length} canvas${canvases.length !== 1 ? 'es' : ''})`;
  scroll.appendChild(header);

  // Grid container
  const grid = document.createElement('div');
  grid.className = 'adhoc-dqm-grid';
  scroll.appendChild(grid);

  canvases.forEach(canvas => {
    const cell = document.createElement('div');
    cell.className = 'adhoc-dqm-cell';

    const label = document.createElement('div');
    label.className = 'adhoc-dqm-label';
    label.textContent = canvas;
    cell.appendChild(label);

    const drawEl = document.createElement('div');
    drawEl.className = 'adhoc-dqm-draw';
    drawEl.id = `adhoc-dqm-draw-${canvas}`;
    cell.appendChild(drawEl);

    grid.appendChild(cell);

    // Fetch and draw
    (async () => {
      const filename = `${base_prefix}_${canvas}.json`;
      try {
        const res = await fetch(`/dqm-output/${filename}?t=${Date.now()}`);
        if (!res.ok) { drawEl.textContent = 'No data'; return; }
        const text = await res.text();
        const jsroot = _getJSROOT();
        const obj = jsroot.parse(text);
        if (!obj) { drawEl.textContent = 'Parse error'; return; }
        await jsroot.draw(drawEl, obj, '');
      } catch (e) {
        drawEl.textContent = `ERR: ${e.message || e}`;
        drawEl.style.cssText = 'color:red;font-size:11px;padding:8px;white-space:pre-wrap;';
      }
    })();
  });

  scroll.scrollTop = scroll.scrollHeight;
}

/* ── Lightbox ──────────────────────────────────────────────── */
function openLightbox(src) {
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox').classList.add('open');
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('open');
}
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    // Close lightbox first, then adhoc popup
    if (document.getElementById('lightbox').classList.contains('open')) {
      closeLightbox();
    } else if (document.getElementById('adhoc-overlay').classList.contains('open')) {
      closeAdhoc();
    }
  }
});

/* ── Voice input (faster-whisper via server) ───────────────── */
let mediaRecorder = null;
let voiceActive = false;

function toggleVoice() {
  voiceActive ? stopVoice() : startVoice();
}

async function startVoice() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    alert('마이크 권한이 필요합니다.');
    return;
  }

  const chunks = [];
  const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
    ? 'audio/webm;codecs=opus' : 'audio/webm';

  mediaRecorder = new MediaRecorder(stream, { mimeType: mime });
  mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };

  mediaRecorder.onstop = async () => {
    stream.getTracks().forEach(t => t.stop());
    const blob = new Blob(chunks, { type: mime });
    const btn  = document.getElementById('mic-btn');
    btn.textContent = '⌛';
    btn.disabled = true;

    try {
      const form = new FormData();
      form.append('audio', blob, 'audio.webm');
      const res  = await fetch('/transcribe', { method: 'POST', body: form });
      const json = await res.json();
      if (json.text) {
        document.getElementById('chat-input').value = json.text;
        sendText();
      }
    } catch (err) {
      console.error('Whisper error:', err);
    } finally {
      btn.textContent = '🎤';
      btn.disabled = false;
    }
  };

  mediaRecorder.start();
  voiceActive = true;
  const btn = document.getElementById('mic-btn');
  btn.textContent = '⏹';
  btn.classList.add('mic-active');
}

function stopVoice() {
  voiceActive = false;
  document.getElementById('mic-btn').classList.remove('mic-active');
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();   // triggers onstop → transcribe
    mediaRecorder = null;
  }
}

/* ── DAQ completion sound ──────────────────────────────────── */
// Place your audio file at:  web/static/audio/daq_complete.*
// Supported formats: mp3, wav, ogg, m4a — browser picks the first
// one it finds.  Falls back to a synthesized chime if no file exists.
const _DAQ_SOUND_SOURCES = [
  '/static/audio/daq_complete.mp3',
  '/static/audio/daq_complete.wav',
  '/static/audio/daq_complete.ogg',
  '/static/audio/daq_complete.m4a',
];

// Pre-load audio element on page load for instant playback.
// Try each source in order; if the browser can't decode it, play()
// will reject and we fall back to the synthesized chime.
const _daqAudio = new Audio(_DAQ_SOUND_SOURCES[0]);
_daqAudio.preload = 'auto';

// Web Audio API fallback (synthesized chime)
let _audioCtx = null;
function _getAudioCtx() {
  if (!_audioCtx)
    _audioCtx = new (window.AudioContext || /** @type {any} */(window).webkitAudioContext)();
  return _audioCtx;
}
// Unlock both AudioContext and HTMLAudioElement on the first user
// interaction. Safari blocks programmatic audio until play() has been
// called inside a user-gesture handler at least once.
function _unlockAudio() {
  const ctx = _getAudioCtx();
  if (ctx.state !== 'running') ctx.resume().catch(() => {});
  _daqAudio.play().then(() => { _daqAudio.pause(); _daqAudio.currentTime = 0; }).catch(() => {});
  document.removeEventListener('click',   _unlockAudio);
  document.removeEventListener('keydown', _unlockAudio);
}
document.addEventListener('click',   _unlockAudio);
document.addEventListener('keydown', _unlockAudio);

function _playChimeFallback() {
  try {
    const ctx = _getAudioCtx();
    const _play = () => {
      try {
        [523.25, 659.25, 783.99].forEach((freq, i) => {
          const osc  = ctx.createOscillator();
          const gain = ctx.createGain();
          osc.connect(gain); gain.connect(ctx.destination);
          osc.type = 'sine'; osc.frequency.value = freq;
          const t = ctx.currentTime + i * 0.15;
          gain.gain.setValueAtTime(0, t);
          gain.gain.linearRampToValueAtTime(0.28, t + 0.02);
          gain.gain.exponentialRampToValueAtTime(0.001, t + 0.45);
          osc.start(t); osc.stop(t + 0.45);
        });
        console.log('[autoTB] DAQ chime played');
      } catch (e) { console.warn('[autoTB] Chime play error:', e); }
    };
    if (ctx.state === 'running') {
      _play();
    } else {
      ctx.resume().then(_play).catch(err => console.warn('[autoTB] Chime resume failed:', err));
    }
  } catch (err) {
    console.warn('[autoTB] Chime failed:', err);
  }
}

// Sound on/off toggle — persisted in localStorage
let _soundEnabled = localStorage.getItem('daqSoundEnabled') !== 'false';

function _updateSoundBtn() {
  const btn = document.getElementById('sound-btn');
  if (btn) btn.textContent = _soundEnabled ? '🔔' : '🔕';
}

function toggleSound() {
  _soundEnabled = !_soundEnabled;
  localStorage.setItem('daqSoundEnabled', _soundEnabled);
  _updateSoundBtn();
}

function playDaqCompleteSound() {
  if (!_soundEnabled) return;
  new Audio(_DAQ_SOUND_SOURCES[0]).play()
    .then(() => console.log('[autoTB] DAQ sound (mp3) played'))
    .catch(err => {
      console.warn('[autoTB] MP3 play failed, falling back to chime:', err.message);
      _playChimeFallback();
    });
}

/* ── DQM live dashboard ─────────────────────────────────────── */
//
// dqm_live_start  → setup grid of cells (manifest-driven), reset state
// dqm_refresh     → re-fetch the JSON for one canvas and JSROOT.redraw it
// dqm_live_end    → mark the title as ended (cells stay visible)
//
// JSROOT is loaded as a module on first use to keep page-load light.

let dqmRun = null;
let dqmBasePrefix = null;
let dqmCells = [];                 // canvas names currently in the grid
const dqmDrawnObjects = new Map(); // canvas name → JSROOT painter (for redraw)
const dqmDrawing = new Set();      // canvases currently mid-draw (prevent concurrent draws)
function _getJSROOT() {
  // jsroot.js is a UMD bundle loaded via <script> tag → registers as window.JSROOT
  if (!window.JSROOT) throw new Error('JSROOT not loaded');
  return window.JSROOT;
}

function dqmCellsContainer() { return document.getElementById('dqm-cells'); }
function dqmTitleEl()        { return document.getElementById('dqm-title'); }

function _emptyDqmCell(message) {
  const div = document.createElement('div');
  div.className = 'dqm-cell empty';
  div.textContent = message;
  return div;
}

function dqmLiveStart(msg) {
  dqmRun = msg.run_number;
  dqmBasePrefix = msg.base_prefix;
  dqmCells = (msg.cells || []).slice();
  dqmDrawnObjects.clear();
  dqmDrawing.clear();

  dqmTitleEl().textContent =
    `DQM Live · Run ${dqmRun} · ${msg.method || ''} · ●LIVE`;

  const cont = dqmCellsContainer();
  cont.innerHTML = '';
  if (dqmCells.length === 0) {
    cont.appendChild(_emptyDqmCell('manifest에 cell이 정의되지 않음'));
    return;
  }
  dqmCells.forEach(addDqmCell);
}

function addDqmCell(canvas) {
  const cont = dqmCellsContainer();
  // De-dupe
  if (cont.querySelector(`[data-canvas="${CSS.escape(canvas)}"]`)) return;

  const card = document.createElement('div');
  card.className = 'dqm-cell';
  card.dataset.canvas = canvas;

  const label = document.createElement('div');
  label.className = 'dqm-cell-label';
  label.textContent = canvas;
  card.appendChild(label);

  const removeBtn = document.createElement('button');
  removeBtn.className = 'dqm-cell-remove';
  removeBtn.textContent = '×';
  removeBtn.title = '제거';
  removeBtn.onclick = (e) => {
    e.stopPropagation();
    dqmCells = dqmCells.filter(c => c !== canvas);
    dqmDrawnObjects.delete(canvas);
    card.remove();
    if (dqmCellsContainer().children.length === 0) {
      dqmCellsContainer().appendChild(_emptyDqmCell('표시할 캔버스가 없습니다 · "+ 캔버스" 로 추가'));
    }
  };
  card.appendChild(removeBtn);

  const draw = document.createElement('div');
  draw.className = 'dqm-cell-draw';
  draw.id = `dqm-draw-${canvas}`;
  card.appendChild(draw);

  card.onclick = () => openDqmModal(canvas);

  // Remove the empty placeholder if present
  const empty = cont.querySelector('.dqm-cell.empty');
  if (empty) empty.remove();

  cont.appendChild(card);

  // Attempt initial draw if a JSON file already exists for this canvas
  _drawCellFromServer(canvas);
}

async function _drawCellFromServer(canvas) {
  if (dqmDrawing.has(canvas)) return;   // 이전 draw 아직 진행 중 → 스킵
  if (!dqmBasePrefix) { console.warn('[DQM] no basePrefix'); return; }
  const filename = `${dqmBasePrefix}_${canvas}.json`;
  const drawEl = document.getElementById(`dqm-draw-${canvas}`);
  if (!drawEl) { console.warn('[DQM] no drawEl for', canvas); return; }
  dqmDrawing.add(canvas);
  try {
    const res = await fetch(`/dqm-output/${filename}?t=${Date.now()}`);
    if (!res.ok) { console.warn('[DQM]', filename, 'HTTP', res.status); return; }
    const text = await res.text();
    const jsroot = _getJSROOT();
    const obj = jsroot.parse(text);
    if (!obj) { console.warn('[DQM] parse returned null for', filename); return; }
    await jsroot.cleanup(drawEl);
    await jsroot.draw(drawEl, obj, '');
    dqmDrawnObjects.set(canvas, obj);
  } catch (e) {
    console.error('[DQM] draw failed', canvas, e);
    drawEl.textContent = `ERR: ${e.message || e}`;
    drawEl.style.cssText = 'color:red;font-size:11px;padding:8px;white-space:pre-wrap;';
  } finally {
    dqmDrawing.delete(canvas);
  }
}

function dqmRefresh(msg) {
  // Only redraw if this canvas is currently shown
  if (!dqmCells.includes(msg.canvas)) return;
  _drawCellFromServer(msg.canvas);
}

function dqmLiveEnd(msg) {
  dqmTitleEl().textContent = `DQM · Run ${msg.run_number} · 종료`;
  // Cells stay so the operator can still browse the final state.
}

/* ── DQM modal (click-to-zoom) ──────────────────────────────── */
async function openDqmModal(canvas) {
  if (!dqmBasePrefix) return;
  document.getElementById('dqm-modal-title').textContent = canvas;
  document.getElementById('dqm-modal').classList.add('open');

  const modalEl = document.getElementById('dqm-modal-draw');
  modalEl.innerHTML = '';

  const filename = `${dqmBasePrefix}_${canvas}.json`;
  try {
    const res = await fetch(`/dqm-output/${filename}?t=${Date.now()}`);
    if (!res.ok) {
      modalEl.textContent = '아직 데이터가 생성되지 않았습니다.';
      return;
    }
    const text = await res.text();
    const jsroot = _getJSROOT();
    const obj = jsroot.parse(text);
    if (!obj) {
      modalEl.textContent = '유효하지 않은 DQM 데이터입니다.';
      return;
    }
    await jsroot.draw(modalEl, obj, '');
  } catch (e) {
    console.error('[DQM] modal draw failed', e);
    modalEl.textContent = '플롯 렌더링 실패: ' + (e.message || e);
    modalEl.style.cssText = 'color:red;font-size:13px;padding:16px;white-space:pre-wrap;';
  }
}

function closeDqmModal() {
  document.getElementById('dqm-modal').classList.remove('open');
  document.getElementById('dqm-modal-draw').innerHTML = '';
}

/* ── DQM "add canvas" picker ────────────────────────────────── */
async function openDqmPicker() {
  if (!dqmRun) {
    alert('DQM live 세션이 아직 시작되지 않았습니다.');
    return;
  }
  document.getElementById('dqm-picker').classList.add('open');
  const list = document.getElementById('dqm-picker-list');
  list.innerHTML = '<div class="dqm-picker-item disabled">불러오는 중…</div>';
  try {
    const res = await fetch(`/api/dqm/canvases/${dqmRun}`);
    const items = await res.json();
    list.innerHTML = '';
    if (!items.length) {
      list.innerHTML = '<div class="dqm-picker-item disabled">아직 생성된 캔버스가 없습니다.</div>';
      return;
    }
    items.forEach(it => {
      const div = document.createElement('div');
      div.className = 'dqm-picker-item';
      const already = dqmCells.includes(it.canvas);
      if (already) div.classList.add('disabled');
      div.innerHTML = `${it.canvas}` +
        `<div class="meta">${it.type} · ${it.method}${already ? ' · 추가됨' : ''}</div>`;
      if (!already) {
        div.onclick = () => {
          dqmCells.push(it.canvas);
          addDqmCell(it.canvas);
          closeDqmPicker();
        };
      }
      list.appendChild(div);
    });
  } catch (e) {
    list.innerHTML = `<div class="dqm-picker-item disabled">에러: ${e.message}</div>`;
  }
}

function closeDqmPicker() {
  document.getElementById('dqm-picker').classList.remove('open');
}

function openDqmFreeform() {
  const url = dqmRun ? `/dqm/freeform?run=${dqmRun}` : '/dqm/freeform';
  window.open(url, '_blank', 'width=1400,height=900');
}

document.addEventListener('DOMContentLoaded', () => {
  const addBtn = document.getElementById('dqm-add-btn');
  const ffBtn  = document.getElementById('dqm-freeform-btn');
  const hvCheckBtn = document.getElementById('hv-check-btn');
  if (addBtn) addBtn.onclick = openDqmPicker;
  if (ffBtn)  ffBtn.onclick  = openDqmFreeform;
  if (hvCheckBtn) hvCheckBtn.onclick = openHvCheck;

  // Initialize empty placeholder
  const cont = document.getElementById('dqm-cells');
  if (cont && cont.children.length === 0) {
    const div = document.createElement('div');
    div.className = 'dqm-cell empty';
    div.textContent = 'DAQ run이 시작되면 자동으로 표시됩니다';
    cont.appendChild(div);
  }
});

// Extend Escape handler to also close DQM modal/picker
document.addEventListener('keydown', e => {
  if (e.key !== 'Escape') return;
  const modal = document.getElementById('dqm-modal');
  const picker = document.getElementById('dqm-picker');
  if (modal && modal.classList.contains('open')) { closeDqmModal(); return; }
  if (picker && picker.classList.contains('open')) { closeDqmPicker(); return; }
});


/* ── Chat input: Enter / Up / Down key handling ─────────────── */
(function () {
  const input = document.getElementById('chat-input');
  if (!input) return;
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      e.preventDefault();
      sendText();
      return;
    }
    if (e.key === 'ArrowUp') {
      if (_cmdHistory.length === 0) return;
      e.preventDefault();
      if (_histIdx === -1) _histDraft = input.value;   // save draft
      _histIdx = Math.min(_histIdx + 1, _cmdHistory.length - 1);
      input.value = _cmdHistory[_histIdx];
      // Move cursor to end
      requestAnimationFrame(() => { input.selectionStart = input.selectionEnd = input.value.length; });
      return;
    }
    if (e.key === 'ArrowDown') {
      if (_histIdx === -1) return;
      e.preventDefault();
      _histIdx -= 1;
      input.value = _histIdx === -1 ? _histDraft : _cmdHistory[_histIdx];
      requestAnimationFrame(() => { input.selectionStart = input.selectionEnd = input.value.length; });
    }
  });
})();

/* ── Tower Picker ──────────────────────────────────────────── */
// 6×6 grid layout (row-major, matches the serpentine image)
const TOWER_GRID = [
  ['M1T1','M1T2','M2T1','M2T2','M3T1','M3T2'],  // row 0
  ['M1T3','M1T4','M2T3','M2T4','M3T3','M3T4'],  // row 1
  ['M4T1','M4T2','M5T1','M5T2','M6T1','M6T2'],  // row 2
  ['M4T3','M4T4','M5T3','M5T4','M6T3','M6T4'],  // row 3
  ['M7T1','M7T2','M8T1','M8T2','M9T1','M9T2'],  // row 4
  ['M7T3','M7T4','M8T3','M8T4','M9T3','M9T4'],  // row 5
];

// Full serpentine order (for sorting selected subset)
const SERPENTINE_ORDER = [
  ...TOWER_GRID[0],
  ...[...TOWER_GRID[1]].reverse(),
  ...TOWER_GRID[2],
  ...[...TOWER_GRID[3]].reverse(),
  ...TOWER_GRID[4],
  ...[...TOWER_GRID[5]].reverse(),
];

let _towerPickerAgent = null;
let _towerPickerMulti = false;
let _towerSelected = new Set();
let _towerDragging = false;
let _towerDragMode = null; // 'select' | 'deselect'

const POSITION_AGENTS = ['position_scan', 'position_scan_sim'];
let _posDirection = null;   // 'horizontal' | 'vertical' | 'both'
let _posChannel = null;     // 'C' | 'S'

function openTowerPicker(agentName) {
  _towerPickerAgent = agentName;
  _towerPickerMulti = !['em_scan', 'position_scan', 'position_scan_sim'].includes(agentName);
  _towerSelected.clear();

  const isPos = POSITION_AGENTS.includes(agentName);
  _posDirection = null;
  _posChannel = null;
  const posOpts = document.getElementById('pos-options');
  if (posOpts) posOpts.style.display = isPos ? 'block' : 'none';
  document.querySelectorAll('.pos-opt-btn').forEach(b => b.classList.remove('selected'));

  const hint = document.getElementById('tower-picker-hint');
  hint.textContent = _towerPickerMulti
    ? '클릭 또는 드래그로 여러 타워 선택'
    : (isPos ? '타워·방향·채널을 선택하세요' : '타워 하나를 선택하세요');

  _buildTowerGrid();
  _updateTowerPickerFooter();
  document.getElementById('tower-picker-overlay').classList.add('open');
}

function selectPosOption(kind, value) {
  if (kind === 'direction') _posDirection = value;
  else if (kind === 'channel') _posChannel = value;
  document.querySelectorAll(`.pos-opt-btn[data-kind="${kind}"]`).forEach(b => {
    b.classList.toggle('selected', b.dataset.value === value);
  });
  _updateTowerPickerFooter();
}

function closeTowerPicker() {
  document.getElementById('tower-picker-overlay').classList.remove('open');
  _towerPickerAgent = null;
}

function clearTowerSelection() {
  _towerSelected.clear();
  document.querySelectorAll('.tower-cell').forEach(c => c.classList.remove('selected'));
  _updateTowerPickerFooter();
}

function _buildTowerGrid() {
  const grid = document.getElementById('tower-grid');
  grid.innerHTML = '';

  TOWER_GRID.forEach(row => {
    row.forEach(tower => {
      const cell = document.createElement('div');
      cell.className = 'tower-cell';
      cell.textContent = tower;
      cell.dataset.tower = tower;

      cell.addEventListener('mousedown', e => {
        e.preventDefault();
        _towerDragging = true;
        const isSelected = _towerSelected.has(tower);
        if (!_towerPickerMulti) {
          // single-select: just toggle to this one
          _towerSelected.clear();
          document.querySelectorAll('.tower-cell').forEach(c => c.classList.remove('selected'));
          _towerSelected.add(tower);
          cell.classList.add('selected');
        } else {
          _towerDragMode = isSelected ? 'deselect' : 'select';
          _toggleTowerCell(cell, tower);
        }
        _updateTowerPickerFooter();
      });

      cell.addEventListener('mouseenter', () => {
        if (!_towerDragging || !_towerPickerMulti) return;
        _toggleTowerCell(cell, tower, _towerDragMode);
        _updateTowerPickerFooter();
      });

      grid.appendChild(cell);
    });
  });

  document.addEventListener('mouseup', () => { _towerDragging = false; }, { once: false });
}

function _toggleTowerCell(cell, tower, forcedMode) {
  const mode = forcedMode || (_towerSelected.has(tower) ? 'deselect' : 'select');
  if (mode === 'select') {
    _towerSelected.add(tower);
    cell.classList.add('selected');
  } else {
    _towerSelected.delete(tower);
    cell.classList.remove('selected');
  }
}

function _updateTowerPickerFooter() {
  const n = _towerSelected.size;
  document.getElementById('tower-picker-count').textContent = `${n}개 선택됨`;
  const confirmBtn = document.getElementById('tower-picker-confirm');
  if (POSITION_AGENTS.includes(_towerPickerAgent)) {
    confirmBtn.disabled = !(n === 1 && _posDirection && _posChannel);
  } else if (_towerPickerMulti) {
    confirmBtn.disabled = (n === 0);
  } else {
    confirmBtn.disabled = (n !== 1);
  }
}

function confirmTowerPicker() {
  if (!_towerPickerAgent) return;
  const agentName = _towerPickerAgent;
  closeTowerPicker();

  // Sort selected towers into serpentine order
  const selected = Array.from(_towerSelected);
  const ordered = SERPENTINE_ORDER.filter(t => selected.includes(t));

  let params = {};
  if (POSITION_AGENTS.includes(agentName)) {
    params = { tower: ordered[0], direction: _posDirection, channel: _posChannel };
  } else if (agentName === 'em_scan') {
    params = { tower: ordered[0] };
  } else {
    params = { tower_order: ordered };
  }

  _launchAgent(agentName, params);
}

/* ── Init ──────────────────────────────────────────────────── */
_updateSoundBtn();
connectWS();
