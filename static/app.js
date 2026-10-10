(() => {
  const form = document.getElementById('chatForm');
  const input = document.getElementById('messageInput');
  const sendButton = document.getElementById('sendButton');
  const messagesNode = document.getElementById('messages');
  const welcome = document.getElementById('welcome');
  const charCount = document.getElementById('charCount');
  const modelStatus = document.getElementById('modelStatus');
  const newChatButton = document.getElementById('newChatButton');
  const clearButton = document.getElementById('clearButton');
  const menuButton = document.getElementById('menuButton');
  const mobileScrim = document.getElementById('mobileScrim');
  const modelDialog = document.getElementById('modelDialog');
  const modelForm = document.getElementById('modelForm');
  const providerSelect = document.getElementById('providerSelect');
  const modelNameInput = document.getElementById('modelNameInput');
  const baseUrlInput = document.getElementById('baseUrlInput');
  const apiKeyInput = document.getElementById('apiKeyInput');
  const apiKeyRow = document.getElementById('apiKeyRow');
  const modelFormMessage = document.getElementById('modelFormMessage');
  const exportChatButton = document.getElementById('exportChatButton');
  const chatSearchInput = document.getElementById('chatSearchInput');
  const chatSearchCount = document.getElementById('chatSearchCount');
  const webGroundingControl = document.getElementById('webGroundingControl');
  const webSearchToggle = document.getElementById('webSearchToggle');
  const chatHistoryToggle = document.getElementById('chatHistoryToggle');

  // Keep the transcript in memory only. Chat prompts are not written to browser storage.
  let transcript = [];
  let sending = false;

  function timeLabel() {
    return new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' }).format(new Date());
  }

  function addMessage(role, text, result = null, promptForActions = null) {
    welcome.hidden = true;
    const article = document.createElement('article');
    article.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'assistant' ? 'A' : 'YOU';
    avatar.setAttribute('aria-hidden', 'true');

    const content = document.createElement('div');
    content.className = 'message-content';
    const head = document.createElement('div');
    head.className = 'message-head';
    const name = document.createElement('span');
    name.textContent = role === 'assistant' ? 'Karlos' : 'You';
    const time = document.createElement('time');
    time.className = 'message-time';
    time.textContent = timeLabel();
    head.append(name, time);

    const body = document.createElement('div');
    body.className = 'message-text';
    body.textContent = text;
    content.append(head, body);

    if (result?.action === 'WITHHOLD' && Array.isArray(result.errors) && result.errors.length) {
      const error = document.createElement('div');
      error.className = 'error-note';
      error.textContent = `Generation issue: ${result.errors[0]}`;
      content.append(error);
    }

    if (result) {
      const chips = document.createElement('div');
      chips.className = 'defense-meta';
      const action = String(result.action || 'REVIEW').toUpperCase();
      const risk = String(result.adversarial_risk || 'LOW').toUpperCase();
      const status = String(result.status || 'UNVERIFIED').replaceAll('_', ' ').toLowerCase();
      addChip(chips, action, action === 'BLOCK' || action === 'WITHHOLD' ? 'block' : action === 'PASS' ? 'pass' : 'warn');
      addChip(chips, `Injection risk: ${risk}`, risk === 'HIGH' ? 'block' : risk === 'MEDIUM' ? 'warn' : 'pass');
      addChip(chips, `Grounding: ${status}`, status === 'verified' ? 'pass' : 'warn');
      if (Number.isFinite(result.latency_ms)) addChip(chips, `${Math.round(result.latency_ms)} ms`, '');
      content.append(chips);

      const audit = document.createElement('details');
      audit.className = 'response-audit';
      const auditSummary = document.createElement('summary');
      auditSummary.textContent = 'Safety & reliability report';
      const auditBody = document.createElement('div');
      auditBody.className = 'audit-details';
      const adv = result.adversarial_details || {};
      const patterns = Array.isArray(adv.detected_patterns) && adv.detected_patterns.length ? adv.detected_patterns.join(', ') : 'No known injection patterns detected';
      addAuditLine(auditBody, 'Injection / jailbreak scan', `${patterns} · ${String(adv.risk_level || 'low').toLowerCase()} risk`);
      addAuditLine(auditBody, 'Model', result.model_backend || 'Active configured model');
      addAuditLine(auditBody, 'Factual risk', String(result.factual_risk || 'unknown').toLowerCase());
      if (Array.isArray(result.web_sources) && result.web_sources.length) {
        addAuditLine(auditBody, 'Web grounding', `${result.web_sources.length} search result(s); ${Number(result.web_crawled_pages || 0)} source page(s) fetched`);
        addAuditLine(auditBody, 'Factuality review', String(result.factuality_review_status || 'not run').replaceAll('_', ' '));
        const sources = document.createElement('div');
        sources.className = 'audit-sources';
        result.web_sources.forEach((source, index) => {
          const row = document.createElement('div');
          row.className = 'audit-source';
          const link = document.createElement('a');
          link.textContent = `[${index + 1}] ${source.title || source.site_name || 'Source'}`;
          try {
            const url = new URL(source.url);
            if (url.protocol === 'https:' || url.protocol === 'http:') {
              link.href = url.href;
              link.target = '_blank';
              link.rel = 'noopener noreferrer';
            }
          } catch {}
          row.append(link);
          if (source.date) {
            const date = document.createElement('span');
            date.textContent = ` · ${source.date}`;
            row.append(date);
          }
          if (source.quarantined === 'true') {
            const warning = document.createElement('span');
            warning.textContent = ' · snippet quarantined';
            warning.className = 'source-warning';
            row.append(warning);
          }
          sources.append(row);
        });
        auditBody.append(sources);
      } else if (result.model_backend?.startsWith('TinyFish')) {
        const groundingStatus = result.web_search_status === 'skipped_by_user'
          ? 'Skipped by your privacy setting; this answer was not web-verified'
          : result.web_search_status === 'failed' ? 'Search failed; answer may be ungrounded' : 'Search completed with no usable results';
        addAuditLine(auditBody, 'Web grounding', groundingStatus);
        addAuditLine(auditBody, 'Factuality review', String(result.factuality_review_status || 'not run').replaceAll('_', ' '));
      }
      addAuditLine(auditBody, 'Evidence coverage', (result.assertions || []).length
        ? `${Math.round((Number(result.evidence_coverage) || 0) * 100)}%`
        : 'Not applicable · no factual claims extracted');
      addAuditLine(auditBody, 'Factual claims checked', `${(result.assertions || []).length} · ${result.corrections?.length || 0} correction(s)`);
      const violations = result.metrics?.output_safety_violations;
      addAuditLine(auditBody, 'Output safety', result.metrics?.output_safety_blocked ? `Blocked${Array.isArray(violations) ? `: ${violations.join(', ')}` : ''}` : 'No configured output-safety violation detected');
      addAuditLine(auditBody, 'Privacy', `${Number(result.metrics?.privacy_redactions || 0)} prompt/context field(s) redacted; generated output is sanitized`);
      addAuditLine(auditBody, 'Tool use', 'Chat does not invoke tools; use the security console sandbox to validate tool calls');
      addAuditLine(auditBody, 'Auditing', `${Number(result.latency_ms || 0).toFixed(0)} ms · ${result.retry_count || 0} retry(s)`);
      if (result.provider_error) addAuditLine(auditBody, 'Search / review detail', result.provider_error);
      if (action === 'WITHHOLD' && result.errors?.length) {
        addAuditLine(auditBody, 'Generation error', result.errors.join(' · '));
      }
      if ((result.assertions || []).length) {
        const claims = document.createElement('ul');
        claims.className = 'audit-claims';
        result.assertions.slice(0, 8).forEach(claim => {
          const item = document.createElement('li');
          item.textContent = `${claim.status}: ${claim.text}`;
          claims.append(item);
        });
        auditBody.append(claims);
      }
      audit.append(auditSummary, auditBody);
      content.append(audit);

      const actions = document.createElement('div');
      actions.className = 'message-actions';
      const copyButton = document.createElement('button');
      copyButton.type = 'button';
      copyButton.className = 'message-action';
      copyButton.textContent = 'Copy';
      copyButton.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(text);
          copyButton.textContent = 'Copied';
          window.setTimeout(() => { copyButton.textContent = 'Copy'; }, 1300);
        } catch {
          copyButton.textContent = 'Copy unavailable';
          window.setTimeout(() => { copyButton.textContent = 'Copy'; }, 1600);
        }
      });
      actions.append(copyButton);
      if (promptForActions) {
        const retryButton = document.createElement('button');
        retryButton.type = 'button';
        retryButton.className = 'message-action';
        retryButton.textContent = '↻ Retry';
        retryButton.addEventListener('click', () => {
          const previousTurns = transcript.slice(0, -2).slice(-6);
          sendMessage(promptForActions, { memoryContext: previousTurns });
        });
        actions.append(retryButton);
      }
      content.append(actions);
    }

    article.append(avatar, content);
    messagesNode.append(article);
    filterMessages();
    window.requestAnimationFrame(() => article.scrollIntoView({ behavior: 'smooth', block: 'nearest' }));
    return article;
  }

  function addChip(parent, label, tone) {
    const chip = document.createElement('span');
    chip.className = `defense-chip ${tone}`.trim();
    chip.textContent = label;
    parent.append(chip);
  }

  function addAuditLine(parent, label, value) {
    const row = document.createElement('div');
    row.className = 'audit-line';
    const title = document.createElement('strong');
    title.textContent = label;
    const text = document.createElement('span');
    text.textContent = String(value);
    row.append(title, text);
    parent.append(row);
  }

  function filterMessages() {
    if (!chatSearchInput) return;
    const query = chatSearchInput.value.trim().toLocaleLowerCase();
    let matches = 0;
    const messages = messagesNode.querySelectorAll('.message:not(.typing-message)');
    messages.forEach(message => {
      const match = !query || message.textContent.toLocaleLowerCase().includes(query);
      message.hidden = !match;
      if (match) matches += 1;
    });
    chatSearchCount.textContent = query ? `${matches} message${matches === 1 ? '' : 's'}` : '';
  }

  function syncExportButton() {
    exportChatButton.disabled = transcript.length === 0;
  }

  function exportChat() {
    if (!transcript.length) return;
    const lines = ['# Karlos chat', '', `Exported ${new Date().toLocaleString()}`, ''];
    transcript.forEach(turn => {
      lines.push(`## ${turn.role === 'user' ? 'You' : 'Karlos'}`, '', turn.text, '');
      if (turn.role === 'assistant' && Array.isArray(turn.sources) && turn.sources.length) {
        lines.push('Sources:');
        turn.sources.forEach(source => {
          if (source.url && /^https?:\/\//i.test(source.url)) {
            lines.push(`- [${source.title || source.site_name || source.url}](${source.url})`);
          }
        });
        lines.push('');
      }
    });
    const blob = new Blob([lines.join('\n')], { type: 'text/markdown;charset=utf-8' });
    const downloadUrl = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = `karlos-chat-${new Date().toISOString().slice(0, 10)}.md`;
    document.body.append(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(downloadUrl);
  }

  function addTyping() {
    const article = addMessage('assistant', '');
    article.classList.add('typing-message');
    const body = article.querySelector('.message-text');
    body.setAttribute('aria-label', 'Karlos is checking your message');
    body.innerHTML = '<span class="typing-indicator" aria-hidden="true"><i></i><i></i><i></i></span><span class="thinking-status">Preparing a protected response · 0s</span>';
    const startedAt = Date.now();
    const status = body.querySelector('.thinking-status');
    article.statusTimer = window.setInterval(() => {
      status.textContent = `Preparing a protected response · ${Math.floor((Date.now() - startedAt) / 1000)}s`;
    }, 1000);
    return article;
  }

  function removeTyping(article) {
    if (article?.statusTimer) window.clearInterval(article.statusTimer);
    article?.remove();
  }

  function resizeInput() {
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
    charCount.textContent = `${input.value.length.toLocaleString()} / 12,000`;
    sendButton.disabled = sending || !input.value.trim();
  }

  async function sendMessage(text = input.value.trim(), options = {}) {
    if (!text || sending) return;
    input.value = '';
    resizeInput();
    closeMobileNav();
    transcript.push({ role: 'user', text });
    syncExportButton();
    addMessage('user', text);
    sending = true;
    sendButton.disabled = true;
    const loading = addTyping();

    try {
      const memoryTurns = chatHistoryToggle.checked
        ? (options.memoryContext ?? transcript.slice(0, -1).slice(-6))
        : [];
      const memory = memoryTurns.map(turn => `${turn.role === 'user' ? 'User' : 'Assistant'}: ${turn.text}`);
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, memory_context: memory.length ? memory : undefined, web_search_enabled: webSearchToggle.checked })
      });
      const data = await response.json().catch(() => ({}));
      removeTyping(loading);
      if (!response.ok) {
        const detail = typeof data.detail === 'string' ? data.detail : `Chat request failed (${response.status}).`;
        addError(detail);
        return;
      }
      const answer = data.final_output || 'The model returned an empty response.';
      transcript.push({ role: 'assistant', text: answer, sources: data.web_sources || [] });
      syncExportButton();
      addMessage('assistant', answer, data, text);
    } catch (error) {
      removeTyping(loading);
      addError('Could not reach the Karlos server. Check that the website server is running and try again.');
    } finally {
      removeTyping(loading);
      sending = false;
      resizeInput();
      input.focus();
    }
  }

  function addError(message) {
    const article = addMessage('assistant', message);
    article.querySelector('.message-text').classList.add('error-note');
  }

  function clearChat() {
    if (sending) return;
    transcript = [];
    messagesNode.replaceChildren();
    chatSearchInput.value = '';
    chatSearchCount.textContent = '';
    syncExportButton();
    welcome.hidden = false;
    input.value = '';
    resizeInput();
    input.focus();
  }

  async function loadStatus() {
    try {
      const response = await fetch('/api/status', { headers: { Accept: 'application/json' } });
      if (!response.ok) throw new Error('status unavailable');
      const data = await response.json();
      modelStatus.textContent = data.model_backend || 'Model connected';
      document.querySelector('.status-orb').style.background = '#39c7a0';
      webGroundingControl.hidden = data.provider !== 'tinyfish';
    } catch {
      modelStatus.textContent = 'Server unavailable';
      document.querySelector('.status-orb').style.background = '#e28b66';
      webGroundingControl.hidden = true;
    }
  }

  function updateProviderFields() {
    const openAI = providerSelect.value === 'openai-compatible';
    const tinyfish = providerSelect.value === 'tinyfish' || providerSelect.value === 'tinyfish-standalone';
    const nvidiaNim = providerSelect.value === 'nvidia-nim';
    const providerNote = document.getElementById('providerNote');
    apiKeyRow.hidden = !openAI;
    if (providerSelect.value === 'ollama') baseUrlInput.value = 'http://localhost:11434';
    else if (openAI && baseUrlInput.value === 'http://localhost:11434') baseUrlInput.value = 'https://api.openai.com/v1';
    else if (nvidiaNim) {
      modelNameInput.value = 'nvidia/nemotron-3.5-lightning-30b-a3b';
      baseUrlInput.value = 'https://integrate.api.nvidia.com/v1';
    }
    modelNameInput.disabled = providerSelect.value === 'demo' || tinyfish || nvidiaNim;
    baseUrlInput.disabled = providerSelect.value === 'demo' || tinyfish || nvidiaNim;
    providerNote.hidden = !tinyfish && !nvidiaNim;
    providerNote.textContent = providerSelect.value === 'tinyfish-standalone'
      ? 'Standalone mode calls TinyFish Search only and displays its results directly. No Ollama or other chat LLM is called. A privacy-redacted query is sent to TinyFish; Karlos still applies its safety checks.'
      : tinyfish
        ? 'The active LLM drafts an answer first. TinyFish then searches and fetches up to three public result pages; the LLM reviews the draft against quarantined page text. Source links appear in the reliability report. The TinyFish key stays on the server.'
        : nvidiaNim
          ? 'Uses the NVIDIA NIM key from the server-side .env file. The model response is held until Karlos completes its safety checks.'
          : '';
  }

  async function openModelSettings() {
    modelFormMessage.textContent = '';
    try {
      const response = await fetch('/api/models');
      const info = await response.json();
      const modelList = document.getElementById('ollamaModels');
      modelList.replaceChildren(...(info.ollama_models || []).map(model => {
        const option = document.createElement('option');
        option.value = model;
        return option;
      }));
      providerSelect.value = info.provider || 'ollama';
      modelNameInput.value = info.model === 'demo' ? 'llama3.1' : (info.model || 'llama3.1');
      baseUrlInput.value = info.base_url || (providerSelect.value === 'ollama' ? 'http://localhost:11434' : 'https://api.openai.com/v1');
    } catch {}
    updateProviderFields();
    modelDialog.hidden = false;
    document.getElementById('modelSettingsButton').setAttribute('aria-expanded', 'true');
  }

  function closeModelSettings() {
    modelDialog.hidden = true;
    document.getElementById('modelSettingsButton').setAttribute('aria-expanded', 'false');
    apiKeyInput.value = '';
  }

  async function saveModelSettings(event) {
    event.preventDefault();
    const button = document.getElementById('saveModelButton');
    button.disabled = true;
    modelFormMessage.textContent = 'Connecting…';
    try {
      const response = await fetch('/api/models', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider: providerSelect.value, model: modelNameInput.value.trim() || 'llama3.1', base_url: baseUrlInput.value.trim(), api_key: apiKeyInput.value })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || 'Could not connect to this model.');
      apiKeyInput.value = '';
      closeModelSettings();
      await loadStatus();
    } catch (error) {
      modelFormMessage.textContent = error.message || 'Could not connect to this model.';
    } finally {
      button.disabled = false;
    }
  }

  function closeMobileNav() {
    document.body.classList.remove('nav-open');
    menuButton.setAttribute('aria-expanded', 'false');
  }

  form.addEventListener('submit', event => {
    event.preventDefault();
    sendMessage();
  });
  input.addEventListener('input', resizeInput);
  input.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });
  document.querySelectorAll('[data-prompt]').forEach(button => {
    button.addEventListener('click', () => sendMessage(button.dataset.prompt));
  });
  newChatButton.addEventListener('click', clearChat);
  clearButton.addEventListener('click', clearChat);
  exportChatButton.addEventListener('click', exportChat);
  chatSearchInput.addEventListener('input', filterMessages);
  menuButton.addEventListener('click', () => {
    const open = document.body.classList.toggle('nav-open');
    menuButton.setAttribute('aria-expanded', String(open));
  });
  mobileScrim.addEventListener('click', closeMobileNav);
  document.getElementById('modelSettingsButton').addEventListener('click', () => {
    if (modelDialog.hidden) openModelSettings();
    else closeModelSettings();
  });
  document.getElementById('closeModelDialog').addEventListener('click', closeModelSettings);
  document.getElementById('cancelModelDialog').addEventListener('click', closeModelSettings);
  providerSelect.addEventListener('change', updateProviderFields);
  modelForm.addEventListener('submit', saveModelSettings);
  document.addEventListener('keydown', event => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
      event.preventDefault();
      clearChat();
    }
    if (event.key === 'Escape') {
      closeMobileNav();
      closeModelSettings();
    }
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'f') {
      event.preventDefault();
      chatSearchInput.focus();
      chatSearchInput.select();
    }
  });

  resizeInput();
  syncExportButton();
  loadStatus();
  window.setInterval(loadStatus, 30_000);
})();
