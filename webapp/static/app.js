const state = {
  messages: [],
  latestRetrieval: null,
  busy: false,
  pendingImage: null,
  pendingImageUrl: "",
  inspectorCollapsed: false,
  sidebarCollapsed: false,
}

const refs = {
  appShell: document.getElementById("appShell"),
  sidebarPanel: document.getElementById("sidebarPanel"),
  toggleSidebarBtn: document.getElementById("toggleSidebarBtn"),
  inspectorPanel: document.getElementById("inspectorPanel"),
  toggleInspectorBtn: document.getElementById("toggleInspectorBtn"),
  messageList: document.getElementById("messageList"),
  chatForm: document.getElementById("chatForm"),
  queryInput: document.getElementById("queryInput"),
  imageInput: document.getElementById("imageInput"),
  uploadPreview: document.getElementById("uploadPreview"),
  sendButton: document.getElementById("sendButton"),
  topKRange: document.getElementById("topKRange"),
  topKValue: document.getElementById("topKValue"),
  debugToggle: document.getElementById("debugToggle"),
  newChatBtn: document.getElementById("newChatBtn"),
  statusCluster: document.getElementById("statusCluster"),
  headerStatus: document.getElementById("headerStatus"),
  inspectorBody: document.getElementById("inspectorBody"),
  inspectorEmpty: document.getElementById("inspectorEmpty"),
  sourceColumns: document.getElementById("sourceColumns"),
  textSources: document.getElementById("textSources"),
  imageSources: document.getElementById("imageSources"),
  debugBox: document.getElementById("debugBox"),
  debugPayload: document.getElementById("debugPayload"),
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;")
}

function formatMs(value) {
  if (!value && value !== 0) return ""
  return `${Math.round(value)}ms`
}

function currentMode() {
  const checked = document.querySelector('input[name="mode"]:checked')
  return checked ? checked.value : window.KB_BOOTSTRAP.defaultMode
}

function setBusy(nextBusy) {
  state.busy = nextBusy
  refs.sendButton.disabled = nextBusy
  refs.newChatBtn.disabled = nextBusy
  refs.imageInput.disabled = nextBusy
  refs.queryInput.disabled = nextBusy
  refs.sendButton.classList.toggle("is-disabled", nextBusy)
}

function autoResizeTextarea() {
  refs.queryInput.style.height = "0px"
  refs.queryInput.style.height = `${Math.min(refs.queryInput.scrollHeight, 220)}px`
}

function resetComposer() {
  refs.queryInput.value = ""
  refs.imageInput.value = ""
  if (state.pendingImageUrl) {
    URL.revokeObjectURL(state.pendingImageUrl)
  }
  state.pendingImage = null
  state.pendingImageUrl = ""
  renderUploadPreview()
  autoResizeTextarea()
}

function renderUploadPreview() {
  if (!state.pendingImage || !state.pendingImageUrl) {
    refs.uploadPreview.classList.add("is-hidden")
    refs.uploadPreview.innerHTML = ""
    refs.chatForm.classList.remove("has-preview")
    return
  }

  refs.chatForm.classList.add("has-preview")
  refs.uploadPreview.classList.remove("is-hidden")
  refs.uploadPreview.innerHTML = `
    <img src="${escapeHtml(state.pendingImageUrl)}" alt="${escapeHtml(state.pendingImage.name)}" />
    <div class="preview-copy">
      <strong>${escapeHtml(state.pendingImage.name)}</strong>
      <small>这张图片会和下一条问题一起发送。</small>
    </div>
    <button class="preview-remove" id="removePreviewBtn" type="button">
      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
    </button>
  `

  document.getElementById("removePreviewBtn").addEventListener("click", () => {
    resetComposer()
  })
}

function renderEmptyState() {
  refs.messageList.innerHTML = `
    <section class="empty-state">
      <div class="section-label">Ready</div>
      <h3>今天想从知识库里得到什么？</h3>
      <p>像通用 AI 对话平台一样直接提问。检索模式、图片输入和调试信息都保留在侧边栏里。</p>
      <ul>
        <li>Hybrid 会同时融合 dense、稀疏、图谱和图片检索。</li>
        <li>上传图片后可以触发多模态检索。</li>
        <li>打开调试信息后，可查看最近一次融合后的召回载荷。</li>
      </ul>
    </section>
  `
}
function renderMessages() {
  if (!state.messages.length) {
    renderEmptyState()
    return
  }

  refs.messageList.innerHTML = state.messages
    .map((message) => {
      const roleLabel = message.role === "human" ? "User" : "Assistant"
      const tags = []
      if (message.latencyMs) tags.push(`<span class="tag">${formatMs(message.latencyMs)}</span>`)
      if (message.rewrittenQuery) tags.push(`<span class="tag">Rewrite: ${escapeHtml(message.rewrittenQuery)}</span>`)
      if (Array.isArray(message.sources)) {
        message.sources.slice(0, 5).forEach((source) => {
          tags.push(`<span class="tag">${source.is_image ? "IMG" : "DOC"} ${escapeHtml(source.name)}</span>`)
        })
      }
      const imageMarkup = message.imageUrl
        ? `<img class="message-image" src="${escapeHtml(message.imageUrl)}" alt="${escapeHtml(message.imageName || "uploaded image")}" />`
        : ""
      const pendingMark = message.pending ? '<span class="tag">Streaming</span>' : ""
      return `
        <article class="message ${message.role === "human" ? "user" : "assistant"}">
          <div class="message-meta">
            <span>${roleLabel}</span>
            ${pendingMark}
          </div>
          <div class="message-body">
            ${imageMarkup}
            <p class="message-content">${escapeHtml(message.content || (message.pending ? "..." : ""))}</p>
          </div>
          ${tags.length ? `<div class="message-tags">${tags.join("")}</div>` : ""}
        </article>
      `
    })
    .join("")

  const last = refs.messageList.lastElementChild
  if (last) {
    last.scrollIntoView({ behavior: "smooth", block: "end" })
  }
}

function renderSources(items, container, imageMode) {
  if (!items.length) {
    container.innerHTML = '<div class="inspector-empty">这个分组里暂时没有结果。</div>'
    return
  }

  container.innerHTML = items
    .map((item) => {
      if (imageMode) {
        const imageMarkup = item.source_url
          ? `<img src="${escapeHtml(item.source_url)}" alt="${escapeHtml(item.source_name)}" />`
          : ""
        return `
          <article class="image-source">
            ${imageMarkup}
            <strong>${escapeHtml(item.source_name)}</strong>
            <p>${escapeHtml(item.content || "图片检索结果")}</p>
            ${item.source_url ? `<a class="source-link" href="${escapeHtml(item.source_url)}" target="_blank" rel="noreferrer">打开文件</a>` : ""}
          </article>
        `
      }

      const summary = (item.content || "").slice(0, 200)
      return `
        <article class="text-source">
          <strong>${escapeHtml(item.source_name)}</strong>
          <p>${escapeHtml(summary)}${item.content && item.content.length > 200 ? "..." : ""}</p>
          ${item.source_url ? `<a class="source-link" href="${escapeHtml(item.source_url)}" target="_blank" rel="noreferrer">打开文件</a>` : ""}
        </article>
      `
    })
    .join("")
}

/* ── Sidebar toggle ── */

function toggleSidebar() {
  state.sidebarCollapsed = !state.sidebarCollapsed
  refs.appShell.classList.toggle("sidebar-left-hidden", state.sidebarCollapsed)
}

function toggleInspector() {
  state.inspectorCollapsed = !state.inspectorCollapsed
  refs.appShell.classList.toggle("sidebar-right-hidden", state.inspectorCollapsed)
}

function renderInspector() {
  if (state.inspectorCollapsed) {
    return
  }

  const retrieval = state.latestRetrieval
  if (!retrieval) {
    refs.inspectorEmpty.style.display = "block"
    refs.sourceColumns.classList.add("is-hidden")
    refs.debugBox.classList.add("is-hidden")
    return
  }

  refs.inspectorEmpty.style.display = "none"
  refs.sourceColumns.classList.remove("is-hidden")

  const textItems = retrieval.items.filter((item) => !item.is_image)
  const imageItems = retrieval.items.filter((item) => item.is_image)
  renderSources(textItems, refs.textSources, false)
  renderSources(imageItems, refs.imageSources, true)

  if (retrieval.debug_info && refs.debugToggle.checked) {
    refs.debugBox.classList.remove("is-hidden")
    refs.debugPayload.textContent = JSON.stringify(retrieval.debug_info, null, 2)
  } else {
    refs.debugBox.classList.add("is-hidden")
    refs.debugPayload.textContent = ""
  }
}

async function refreshHealth() {
  try {
    const response = await fetch("/api/health")
    const payload = await response.json()
    const rows = [
      ["Qdrant", payload.qdrant],
      ["Neo4j", payload.neo4j],
      ["Ready", payload.ready],
    ]

    refs.statusCluster.innerHTML = rows
      .map(([label, ok]) => `
        <div class="status-pill">
          <span class="status-dot ${ok ? "is-on" : "is-off"}"></span>
          <span>${label}</span>
        </div>
      `)
      .join("")

    if (payload.ready) {
      refs.headerStatus.textContent = "KB Assistant"
    } else {
      refs.headerStatus.textContent = "KB Assistant — 部分服务不可用"
    }
  } catch (_) {
    refs.headerStatus.textContent = "KB Assistant — 健康检查失败"
  }
}

function parseSseBlock(block) {
  const lines = block.split("\n")
  let eventName = "message"
  const dataLines = []

  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim()
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim())
    }
  }

  if (!dataLines.length) return null
  return { eventName, payload: JSON.parse(dataLines.join("\n")) }
}
async function consumeStream(response, assistantMessage) {
  const reader = response.body.getReader()
  const decoder = new TextDecoder("utf-8")
  let buffer = ""

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const chunks = buffer.split("\n\n")
    buffer = chunks.pop() || ""

    for (const chunk of chunks) {
      const parsed = parseSseBlock(chunk)
      if (!parsed) continue

      if (parsed.eventName === "meta") {
        assistantMessage.rewrittenQuery = parsed.payload.rewritten_query || ""
      }

      if (parsed.eventName === "token") {
        assistantMessage.content += parsed.payload.delta || ""
        renderMessages()
      }

      if (parsed.eventName === "error") {
        throw new Error(parsed.payload.message || "Stream failed")
      }

      if (parsed.eventName === "done") {
        assistantMessage.content = parsed.payload.answer || assistantMessage.content
        assistantMessage.pending = false
        assistantMessage.rewrittenQuery = parsed.payload.rewritten_query || assistantMessage.rewrittenQuery
        assistantMessage.latencyMs = parsed.payload.retrieval?.latency_ms || null
        assistantMessage.sources = parsed.payload.retrieval?.sources || []
        state.latestRetrieval = parsed.payload.retrieval || null
        renderMessages()
        renderInspector()
      }
    }
  }
}

async function handleSubmit(event) {
  event.preventDefault()
  if (state.busy) return

  const query = refs.queryInput.value.trim()
  if (!query) return

  const imageUrl = state.pendingImageUrl
  const imageName = state.pendingImage ? state.pendingImage.name : ""

  state.messages.push({
    role: "human",
    content: query,
    imageUrl,
    imageName,
  })

  const assistantMessage = {
    role: "ai",
    content: "",
    pending: true,
    latencyMs: null,
    rewrittenQuery: "",
    sources: [],
  }
  state.messages.push(assistantMessage)
  renderMessages()
  setBusy(true)

  const formData = new FormData()
  formData.append("query", query)
  formData.append("mode", currentMode())
  formData.append("top_k", refs.topKRange.value)
  formData.append("debug", String(refs.debugToggle.checked))
  formData.append(
    "chat_history",
    JSON.stringify(
      state.messages
        .slice(0, -2)
        .map((message) => ({ role: message.role, content: message.content }))
    )
  )

  if (state.pendingImage) {
    formData.append("image", state.pendingImage)
  }

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      body: formData,
    })

    if (!response.ok || !response.body) {
      const text = await response.text()
      throw new Error(text || `HTTP ${response.status}`)
    }

    resetComposer()
    await consumeStream(response, assistantMessage)
  } catch (error) {
    assistantMessage.pending = false
    assistantMessage.content = `请求失败：${error.message}`
    renderMessages()
  } finally {
    setBusy(false)
  }
}

/* ── Event bindings ── */

refs.chatForm.addEventListener("submit", handleSubmit)
refs.imageInput.addEventListener("change", (event) => {
  const [file] = event.target.files || []
  if (!file) {
    resetComposer()
    return
  }

  if (state.pendingImageUrl) {
    URL.revokeObjectURL(state.pendingImageUrl)
  }

  state.pendingImage = file
  state.pendingImageUrl = URL.createObjectURL(file)
  renderUploadPreview()
})
refs.topKRange.addEventListener("input", () => {
  refs.topKValue.textContent = refs.topKRange.value
})
refs.debugToggle.addEventListener("change", renderInspector)
refs.queryInput.addEventListener("input", autoResizeTextarea)
refs.newChatBtn.addEventListener("click", () => {
  state.messages = []
  state.latestRetrieval = null
  resetComposer()
  renderMessages()
  renderInspector()
})

refs.toggleSidebarBtn.addEventListener("click", toggleSidebar)
refs.toggleInspectorBtn.addEventListener("click", toggleInspector)

/* ── Init ── */

renderMessages()
renderInspector()
autoResizeTextarea()
refreshHealth()
setInterval(refreshHealth, 30000)
