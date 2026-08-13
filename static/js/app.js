  const form = document.getElementById('chat-form');
  const textarea = document.getElementById('query');
  const historyEl = document.getElementById('history');
  const chatArea = document.getElementById('chat-area');
  const sendBtn = document.getElementById('send-btn');
  const errorBox = document.getElementById('error');
  const convListEl = document.getElementById('conv-list');
  const newChatBtn = document.getElementById('new-chat-btn');
  const attachBtn = document.getElementById('attach-btn');
  const helpBtn = document.getElementById('help-btn');
  const imageInput = document.getElementById('image-input');
  const attachmentPreviewEl = document.getElementById('attachment-preview');
  const attachmentThumbEl = attachmentPreviewEl.querySelector('.attachment-thumb');
  const attachmentNameEl = attachmentPreviewEl.querySelector('.attachment-name');
  const attachmentRemoveEl = attachmentPreviewEl.querySelector('.attachment-remove');
  const helpModeIndicatorEl = document.getElementById('help-mode-indicator');
  const helpModeCancelEl = helpModeIndicatorEl.querySelector('.help-mode-cancel');

  let currentConversationId = localStorage.getItem('currentConversationId');
  currentConversationId = currentConversationId ? parseInt(currentConversationId, 10) : null;

  // An image picked via the attach button, staged until the next send.
  // { dataUrl, base64, mimeType, filename } or null.
  let pendingImage = null;

  // Whether the ❓ help-mode toggle is armed for the NEXT message. This is
  // deliberately just a flag, not a canned message: the whole point (per
  // explicit feedback on the first version of this feature) is that the
  // user types their OWN question — which may be about anything, not just
  // astrology — and it gets force-routed to astro_help_assistant
  // (ChatRequest.force_help, routes/chat.py) instead of going through
  // tool_router's own classification, which is exactly the least reliable
  // for a user who doesn't know what to ask for yet. One-shot: consumed
  // by the next send, same lifecycle as pendingImage above.
  let helpModeArmed = false;

  function setHelpMode(active) {
    helpModeArmed = active;
    helpBtn.classList.toggle('active', active);
    helpModeIndicatorEl.style.display = active ? 'flex' : 'none';
  }

  // Polling for background-resolved messages (e.g. pending image jobs).
  const POLL_INTERVAL_MS = 5000;
  let pollTimer = null;

  // ---------- Форматирование ----------
  function fmtTime(ms) {
    return new Date(ms).toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }
  function fmtDuration(ms) {
    return (ms / 1000).toFixed(1) + ' с';
  }
  function fmtSize(bytes) {
    if (bytes < 1024) return bytes + ' Б';
    return (bytes / 1024).toFixed(1) + ' КБ';
  }

  // ---------- Копирование в буфер обмена ----------
  async function copyTextToClipboard(text) {
    // navigator.clipboard requires a secure context (https, or localhost) —
    // the app is often opened over plain http from a LAN address, where
    // that API is unavailable, so fall back to the older execCommand trick
    // in that case instead of silently failing.
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return;
    }
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    try {
      if (!document.execCommand('copy')) {
        throw new Error('execCommand copy failed');
      }
    } finally {
      document.body.removeChild(ta);
    }
  }

  function makeCopyButton(getText, extraClass, title) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'copy-btn' + (extraClass ? ' ' + extraClass : '');
    btn.title = title || 'Скопировать';
    btn.textContent = '📋';
    btn.addEventListener('click', async (ev) => {
      ev.stopPropagation();
      try {
        await copyTextToClipboard(getText());
        btn.textContent = '✓';
        btn.classList.add('copied');
      } catch (err) {
        btn.textContent = '⚠';
      }
      setTimeout(() => {
        btn.textContent = '📋';
        btn.classList.remove('copied');
      }, 1500);
    });
    return btn;
  }

  function makePdfExportButton(messageId) {
    // Plain link styled as a button (not a click handler that fetches and
    // blob-downloads) — GET /api/messages/{id}/pdf serves
    // Content-Disposition: inline (see routes/export.py), so opening it
    // in a new tab shows the PDF in the browser's own viewer, which
    // already has its own save/print controls; no extra JS needed.
    //
    // Text label, not an emoji: 📄 (page facing up) used to be reused here
    // AND for the generic file-attachment icon (renderFileCard below),
    // making the two visually indistinguishable despite meaning different
    // things — "PDF" reads unambiguously regardless of platform emoji
    // rendering.
    const btn = document.createElement('a');
    btn.className = 'copy-btn pdf-export-btn';
    btn.title = 'Экспортировать в PDF';
    btn.textContent = 'PDF';
    btn.href = '/api/messages/' + messageId + '/pdf';
    btn.target = '_blank';
    btn.rel = 'noopener';
    return btn;
  }

  // ---------- Рендер сообщений ----------
  //
  // renderProseText below is a deliberately tiny "markdown-lite" renderer,
  // not a full markdown parser and not raw HTML from the model. It exists
  // because the astro answer prompt (utils/interpret.py) asks the model
  // for "## Заголовок" section headers and "**жирный**" emphasis, and this
  // UI used to just dump that text via textContent — so those markers
  // showed up as literal "##"/"**" characters instead of being rendered.
  // Two ways to fix that were considered: (a) have the model emit raw
  // HTML tags directly and drop them into innerHTML as-is, or (b) keep the
  // model's existing markdown-style output and parse just the handful of
  // patterns actually used into DOM elements built via textContent/
  // createElement. (b) is what's implemented here — it never calls
  // innerHTML with model-generated text at all, so any stray "<", ">", or
  // "&" the model happens to write (degree comparisons, HTML-like
  // abbreviations, whatever) always renders as plain literal text instead
  // of being interpreted as markup — safe by construction rather than by
  // sanitizing, and it needed no prompt change since the model was already
  // emitting correct markdown syntax.
  function renderInlineFormatted(el, text) {
    // Splits on **bold** spans; everything else (including any literal
    // "<"/">"/"&") goes through as plain text nodes, never interpreted as
    // markup. Single newlines within a paragraph become <br> so multi-line
    // list-like content (e.g. "1. ...\n2. ...") still breaks visually.
    const parts = text.split(/(\*\*[^*\n]+\*\*)/g);
    for (const part of parts) {
      if (part.length > 4 && part.startsWith('**') && part.endsWith('**')) {
        const strong = document.createElement('strong');
        strong.textContent = part.slice(2, -2);
        el.appendChild(strong);
        continue;
      }
      const lines = part.split('\n');
      lines.forEach((line, i) => {
        if (i > 0) el.appendChild(document.createElement('br'));
        if (line) el.appendChild(document.createTextNode(line));
      });
    }
  }

  function renderProseText(container, text) {
    // Line-by-line, not paragraph-first: a real answer often puts a
    // "## Heading" directly above the very next line of prose with no
    // blank line in between (the model follows the prompt's "## Section"
    // instruction literally but doesn't always add a blank line after
    // it) — splitting on blank lines first would leave a heading line
    // glued to the following paragraph as one blob, so the "#" marker
    // never gets recognized as its own heading at all. Scanning line by
    // line and flushing the accumulated paragraph buffer whenever a
    // heading line or a blank line is hit fixes that (confirmed against
    // a real generated answer with this exact heading-then-prose shape).
    const lines = text.split('\n');
    let buffer = [];
    const flush = () => {
      if (!buffer.length) return;
      const para = buffer.join('\n').trim();
      buffer = [];
      if (!para) return;
      const p = document.createElement('div');
      p.className = 'msg-text';
      renderInlineFormatted(p, para);
      container.appendChild(p);
    };
    for (const line of lines) {
      if (line.trim() === '') {
        flush();
        continue;
      }
      // ## -> h3, since the message bubble itself is already the "page" —
      // starting one level below a top-level h2 reads better than h1.
      const headingMatch = line.match(/^(#{1,6})\s+(.+)$/);
      if (headingMatch) {
        flush();
        const level = Math.min(6, headingMatch[1].length + 2);
        const h = document.createElement('h' + level);
        h.className = 'msg-heading';
        renderInlineFormatted(h, headingMatch[2].trim());
        container.appendChild(h);
        continue;
      }
      buffer.push(line);
    }
    flush();
  }

  function renderMessageBody(container, text) {
    const fenceRe = /```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```/g;
    let lastIndex = 0;
    let match;
    let any = false;

    while ((match = fenceRe.exec(text)) !== null) {
      any = true;
      const before = text.slice(lastIndex, match.index).trim();
      if (before) {
        renderProseText(container, before);
      }
      const pre = document.createElement('pre');
      pre.className = 'code-block';
      const code = document.createElement('code');
      code.textContent = match[2].replace(/\n$/, '');
      pre.appendChild(code);
      pre.appendChild(makeCopyButton(() => code.textContent, 'code-copy-btn', 'Скопировать код'));
      container.appendChild(pre);
      lastIndex = fenceRe.lastIndex;
    }

    const rest = text.slice(lastIndex);
    if (!any) {
      renderProseText(container, text);
    } else if (rest.trim()) {
      renderProseText(container, rest.trim());
    }
  }

  function isImageFile(f) {
    return typeof f.mime_type === 'string' && f.mime_type.startsWith('image/');
  }

  function renderImageAttachment(f) {
    const wrap = document.createElement('div');
    wrap.className = 'image-attachment';
    const img = document.createElement('img');
    // f._localSrc is set only for the optimistic preview of an image the
    // user just picked, before the server has assigned it a file id (see
    // the submit handler below); every other case (including this same
    // message after a page reload) uses the normal /api/files/{id} URL.
    const src = f._localSrc || ('/api/files/' + f.id);
    img.src = src;
    img.alt = f.filename;
    // Wrapping in a link so the image can be opened full-size in its own
    // tab (e.g. for a wheel chart's fine detail) — previously there was
    // no way to do this at all, only the inline thumbnail. Only wrap real
    // server-side attachments (an optimistic local preview has no server
    // URL yet to open). routes/files.py now serves image/* with
    // Content-Disposition: inline, so this opens as a normal viewable
    // page instead of triggering a download.
    if (f._localSrc) {
      wrap.appendChild(img);
    } else {
      const link = document.createElement('a');
      link.href = src;
      link.target = '_blank';
      link.rel = 'noopener';
      link.title = 'Открыть в новом окне';
      link.appendChild(img);
      wrap.appendChild(link);
    }
    return wrap;
  }

  function renderFileCard(f) {
    const card = document.createElement('div');
    card.className = 'file-card';

    const icon = document.createElement('span');
    icon.className = 'icon';
    icon.textContent = '📄';

    const link = document.createElement('a');
    link.href = '/api/files/' + f.id;
    link.textContent = f.filename;
    link.setAttribute('download', f.filename);

    const meta = document.createElement('span');
    meta.className = 'meta';
    meta.textContent = fmtSize(f.size);

    card.appendChild(icon);
    card.appendChild(link);
    card.appendChild(meta);
    return card;
  }

  function addMessageFromRecord(record) {
    const wrap = document.createElement('div');
    const statusClass = record.status === 'pending' ? ' pending' : record.status === 'error' ? ' error' : '';
    wrap.className = 'msg-wrap ' + (record.role === 'user' ? 'user' : 'bot') + statusClass;

    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    // Wheel charts (_attach_chart_if_applicable in routes/chat.py) are
    // always saved as image/svg+xml — every other image attachment
    // (a generated/edited picture from ycplt_img, or a user's own upload)
    // is a raster type (image/png etc.), so this mime check is a reliable
    // way to single out chart images without any extra flag from the
    // backend. Rendered before the text body, per user feedback wanting
    // the chart to lead the post instead of trailing after the text.
    const files = record.files || [];
    const chartFiles = files.filter((f) => f.mime_type === 'image/svg+xml');
    const otherFiles = files.filter((f) => f.mime_type !== 'image/svg+xml');
    for (const f of chartFiles) {
      bubble.appendChild(renderImageAttachment(f));
    }

    renderMessageBody(bubble, record.content || '');

    if (record.content && record.content.trim()) {
      bubble.appendChild(makeCopyButton(() => record.content, 'msg-copy-btn', 'Скопировать сообщение'));
    }

    // PDF export needs a real DB message id — absent for the pending
    // placeholder shown while an image job runs, and would 404 if clicked
    // then, so it's only added once the message has settled into a real
    // status. A message loaded from history (page reload/switch) always
    // has record.id already; a message just received from POST /chat has
    // it too now (data.message_id — see routes/chat.py's response dicts).
    if (record.id && record.status !== 'pending') {
      bubble.appendChild(makePdfExportButton(record.id));
    }

    for (const f of otherFiles) {
      if (isImageFile(f)) {
        bubble.appendChild(renderImageAttachment(f));
      } else {
        bubble.appendChild(renderFileCard(f));
      }
    }
    wrap.appendChild(bubble);

    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    if (record.role === 'user') {
      meta.textContent = 'Отправлено ' + fmtTime(record.created_at);
    } else if (record.status === 'pending') {
      meta.textContent = 'Ожидание результата…';
    } else if (record.status === 'error') {
      meta.textContent = 'Ошибка · ' + fmtTime(record.created_at);
    } else {
      let text = 'Ответ ' + fmtTime(record.created_at);
      if (record.thinking_ms != null) {
        text += ' · думал ' + fmtDuration(record.thinking_ms);
      }
      meta.textContent = text;
    }
    wrap.appendChild(meta);

    historyEl.appendChild(wrap);
    chatArea.scrollTop = chatArea.scrollHeight;
    return wrap;
  }

  function addPending() {
    const wrap = document.createElement('div');
    wrap.className = 'msg-wrap bot pending';
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = 'Модель думает...';
    wrap.appendChild(bubble);
    historyEl.appendChild(wrap);
    chatArea.scrollTop = chatArea.scrollHeight;
    return wrap;
  }

  // ---------- Диалоги (сайдбар) ----------
  async function loadConversations() {
    const resp = await fetch('/api/conversations');
    const list = await resp.json();
    renderSidebar(list);
    return list;
  }

  function renderSidebar(list) {
    convListEl.innerHTML = '';
    if (!list.length) {
      const empty = document.createElement('div');
      empty.id = 'sidebar-empty';
      empty.textContent = 'Пока нет ни одного чата';
      convListEl.appendChild(empty);
      return;
    }
    for (const conv of list) {
      const item = document.createElement('div');
      item.className = 'conv-item' + (conv.id === currentConversationId ? ' active' : '');

      const title = document.createElement('span');
      title.className = 'conv-title';
      title.textContent = conv.title || 'Новый чат';

      const ren = document.createElement('button');
      ren.className = 'ren-btn';
      ren.type = 'button';
      ren.textContent = '✎';
      ren.title = 'Переименовать чат';
      ren.addEventListener('click', (ev) => {
        ev.stopPropagation();
        startRenameConversation(item, title, conv);
      });

      const dl = document.createElement('button');
      dl.className = 'dl-btn';
      dl.type = 'button';
      dl.textContent = '⬇';
      dl.title = 'Скачать архив чата (дамп + вложения)';
      dl.addEventListener('click', (ev) => {
        ev.stopPropagation();
        window.location.href = '/api/conversations/' + conv.id + '/export';
      });

      const del = document.createElement('button');
      del.className = 'del-btn';
      del.type = 'button';
      del.textContent = '✕';
      del.title = 'Удалить чат';
      del.addEventListener('click', (ev) => deleteConversation(conv.id, ev));

      item.appendChild(title);
      item.appendChild(ren);
      item.appendChild(dl);
      item.appendChild(del);
      item.addEventListener('click', () => selectConversation(conv.id));
      convListEl.appendChild(item);
    }
  }

  // Swaps the .conv-title span for a text input in place, saves via PATCH
  // on Enter/blur, discards on Escape. A dedicated ✎ button (rather than
  // e.g. double-click on the title) on purpose — the title text is already
  // the row's primary click target for opening the chat, so overloading it
  // with a second gesture would race against that existing click handler
  // (both fire on the same element) instead of just being a separate,
  // unambiguous control.
  function startRenameConversation(item, titleEl, conv) {
    const input = document.createElement('input');
    input.className = 'conv-title-input';
    input.type = 'text';
    input.value = conv.title || '';
    item.replaceChild(input, titleEl);
    input.focus();
    input.select();

    // Prevents a click inside the input (e.g. to move the caret) from
    // bubbling up to the row's own click handler and switching to this
    // conversation mid-edit.
    input.addEventListener('click', (ev) => ev.stopPropagation());

    let cancelled = false;
    let finished = false;

    const finish = async () => {
      if (finished) return;
      finished = true;
      const newTitle = input.value.trim();
      if (!cancelled && newTitle && newTitle !== conv.title) {
        try {
          await fetch('/api/conversations/' + conv.id, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: newTitle }),
          });
        } catch (e) {
          // Network error — fall through to reload; the title just stays
          // whatever it was on the server since the rename didn't happen.
        }
      }
      await loadConversations();
    };

    input.addEventListener('blur', finish);
    input.addEventListener('keydown', (ev) => {
      ev.stopPropagation();
      if (ev.key === 'Enter') {
        ev.preventDefault();
        input.blur();
      } else if (ev.key === 'Escape') {
        cancelled = true;
        input.blur();
      }
    });
  }

  function stopPolling() {
    if (pollTimer) {
      clearTimeout(pollTimer);
      pollTimer = null;
    }
  }

  function schedulePoll(id) {
    stopPolling();
    pollTimer = setTimeout(() => refreshConversation(id), POLL_INTERVAL_MS);
  }

  async function refreshConversation(id) {
    // Re-fetches messages for the conversation that's still open and, if any
    // message is still pending (e.g. an image job still in progress on
    // ycplt_img), schedules another check. Stops once nothing is pending.
    if (id !== currentConversationId) return;

    const resp = await fetch(`/api/conversations/${id}/messages`);
    if (!resp.ok) return;
    const messages = await resp.json();

    historyEl.innerHTML = '';
    for (const m of messages) {
      addMessageFromRecord(m);
    }

    if (messages.some((m) => m.status === 'pending')) {
      schedulePoll(id);
    }
  }

  async function selectConversation(id) {
    currentConversationId = id;
    localStorage.setItem('currentConversationId', id);
    stopPolling();

    const resp = await fetch(`/api/conversations/${id}/messages`);
    historyEl.innerHTML = '';
    let messages = [];
    if (resp.ok) {
      messages = await resp.json();
      for (const m of messages) {
        addMessageFromRecord(m);
      }
    }
    await loadConversations();

    if (messages.some((m) => m.status === 'pending')) {
      schedulePoll(id);
    }
  }

  function newChat() {
    stopPolling();
    currentConversationId = null;
    localStorage.removeItem('currentConversationId');
    historyEl.innerHTML = '';
    errorBox.textContent = '';
    loadConversations();
    textarea.focus();
  }

  async function deleteConversation(id, ev) {
    ev.stopPropagation();
    if (!confirm('Удалить этот чат?')) return;
    await fetch('/api/conversations/' + id, { method: 'DELETE' });
    if (id === currentConversationId) {
      newChat();
    } else {
      await loadConversations();
    }
  }

  newChatBtn.addEventListener('click', newChat);

  // ---------- Вложение изображения (для редактирования) ----------
  function renderAttachmentPreview() {
    if (!pendingImage) {
      attachmentPreviewEl.style.display = 'none';
      attachmentThumbEl.src = '';
      attachmentNameEl.textContent = '';
      return;
    }
    attachmentPreviewEl.style.display = 'flex';
    attachmentThumbEl.src = pendingImage.dataUrl;
    attachmentNameEl.textContent = pendingImage.filename;
  }

  function clearAttachedImage() {
    pendingImage = null;
    imageInput.value = '';
    renderAttachmentPreview();
  }

  function handleImageSelected(file) {
    if (!file || !file.type || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result;
      const base64 = String(dataUrl).split(',')[1] || '';
      pendingImage = {
        dataUrl,
        base64,
        mimeType: file.type || 'image/png',
        filename: file.name || 'upload.png',
      };
      renderAttachmentPreview();
    };
    reader.readAsDataURL(file);
  }

  attachBtn.addEventListener('click', () => imageInput.click());
  imageInput.addEventListener('change', () => {
    handleImageSelected(imageInput.files && imageInput.files[0]);
  });
  attachmentRemoveEl.addEventListener('click', clearAttachedImage);

  // ---------- Режим помощи ----------
  // Arms/disarms help-mode for the NEXT message only — see helpModeArmed's
  // own comment above. Nothing is sent here; the user still types and
  // sends their own message as usual, on any topic.
  helpBtn.addEventListener('click', () => setHelpMode(!helpModeArmed));
  helpModeCancelEl.addEventListener('click', () => setHelpMode(false));

  // ---------- Отправка сообщения ----------
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = textarea.value.trim();
    if (!query) return;

    // Captured before clearAttachedImage()/setHelpMode(false) reset the
    // composer, so the request still reflects what the user actually
    // armed/attached for THIS message. sentForConversationId is captured
    // for the same reason but a different purpose: a request can take
    // minutes (a full RAG-interpreted astro answer), and the backend
    // keeps computing it regardless of what's on screen by the time it
    // finishes — nothing cancels it just because the user switched to
    // another chat. Without this, the response handler below used to
    // unconditionally overwrite currentConversationId and splice the
    // reply into whatever's currently displayed, which could inject one
    // conversation's answer into a completely different, currently-open
    // one. Comparing against this captured id at response time is what
    // fixes that — see below.
    const imageToSend = pendingImage;
    const forceHelp = helpModeArmed;
    const sentForConversationId = currentConversationId;

    errorBox.textContent = '';
    textarea.value = '';
    sendBtn.disabled = true;
    setHelpMode(false); // one-shot — consumed by this message

    const userRecord = { role: 'user', content: query, created_at: Date.now() };
    if (imageToSend) {
      // No server file id yet — renderImageAttachment() falls back to
      // _localSrc (a data: URL) for this one render; after a reload the
      // same attachment comes from the server with a real id and uses the
      // normal /api/files/{id} URL instead.
      userRecord.files = [{
        id: null,
        filename: imageToSend.filename,
        mime_type: imageToSend.mimeType,
        size: imageToSend.base64.length,
        _localSrc: imageToSend.dataUrl,
      }];
    }
    addMessageFromRecord(userRecord);
    clearAttachedImage();
    const pending = addPending();

    try {
      const body = { query, conversation_id: currentConversationId };
      if (imageToSend) {
        body.image_data = imageToSend.base64;
        body.image_filename = imageToSend.filename;
        body.image_mime_type = imageToSend.mimeType;
      }
      if (forceHelp) {
        body.force_help = true;
      }
      const resp = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const data = await resp.json();
      pending.remove();

      // True only if the user is still looking at the same chat this
      // request was sent from (or, for a brand-new chat, hasn't since
      // switched to some OTHER existing conversation) — see
      // sentForConversationId's own comment above for why this matters.
      // A brand-new chat has no id to compare until the server assigns
      // one, so that case is judged by "still on the new-chat screen"
      // instead.
      const stillOnSameChat = sentForConversationId === null
        ? currentConversationId === null
        : currentConversationId === sentForConversationId;

      if (!resp.ok) {
        if (stillOnSameChat) {
          errorBox.textContent = data.detail || 'Ошибка запроса';
        }
      } else {
        // The reply is already persisted server-side under
        // data.conversation_id regardless of what's on screen — refresh
        // the sidebar unconditionally so it shows up there (new title/
        // updated_at) even if the user has moved on to another chat.
        await loadConversations();

        if (stillOnSameChat) {
          currentConversationId = data.conversation_id;
          localStorage.setItem('currentConversationId', currentConversationId);
          addMessageFromRecord({
            id: data.message_id,
            role: 'assistant',
            content: data.response,
            created_at: data.responded_at,
            thinking_ms: data.thinking_ms,
            status: data.status,
            files: data.files
          });

          // An image request returns status "pending" immediately (the job
          // runs on ycplt_img in the background). Start polling right away so
          // the placeholder gets replaced with the image once it's ready,
          // instead of waiting for the user to reload or switch chats.
          if (data.status === 'pending') {
            schedulePoll(currentConversationId);
          }
        }
        // else: the user switched to a different chat while this was in
        // flight. Its answer stays safely in the database under its own
        // conversation — opening that conversation (now or later) loads
        // it from the server as normal; we just must not force-switch the
        // UI back to it or splice its content into whatever's open now.
      }
    } catch (err) {
      pending.remove();
      if (sentForConversationId === null ? currentConversationId === null : currentConversationId === sentForConversationId) {
        errorBox.textContent = 'Не удалось связаться с сервером: ' + err;
      }
    } finally {
      sendBtn.disabled = false;
      textarea.focus();
    }
  });

  textarea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  // ---------- Инициализация ----------
  (async function init() {
    const list = await loadConversations();
    if (currentConversationId && list.some(c => c.id === currentConversationId)) {
      await selectConversation(currentConversationId);
    } else if (list.length > 0) {
      await selectConversation(list[0].id);
    } else {
      currentConversationId = null;
    }
  })();
