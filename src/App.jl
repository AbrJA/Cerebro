module App

using Genie
using Genie.Router
using Genie.Renderer.Html
using Genie.Renderer.Json
using Genie.Requests

using ..Ingestion
using ..Backend

const DB_PATH = "vector_db.json"
const GLOBAL_DB = Ref{Tuple{Vector{ParentChunk},Vector{ChildChunk},BinaryIndex}}()

function init_db(folder_path::String)
    if isfile(DB_PATH)
        println("=> Loading existing database from $DB_PATH...")
        parents, children, index = load_database(DB_PATH)
        GLOBAL_DB[] = (parents, children, index)
    else
        println("=> No database found. Building from $folder_path...")
        mkpath(folder_path)
        docs = load_documents(folder_path)
        if isempty(docs)
            println("=> No documents found in $folder_path. Please add some .txt or .md files to test.")
            GLOBAL_DB[] = (ParentChunk[], ChildChunk[], BinaryIndex(Vector{Int8}[], 0))
            return
        end
        println("=> Chunking $(length(docs)) documents...")
        parents, children = hierarchical_chunk(docs)
        println("=> Embedding and quantizing $(length(children)) child chunks...")
        index = embed_and_quantize!(children)
        println("=> Saving database...")
        save_database(DB_PATH, parents, children, index)
        GLOBAL_DB[] = (parents, children, index)
    end
    println("=> Database loaded: $(length(GLOBAL_DB[][1])) parents, $(length(GLOBAL_DB[][2])) children, $(length(GLOBAL_DB[][3].vectors)) binary vectors.")
end

function chat_ui()
    html("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Cerebro RAG Chat</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #3b82f6;
                --primary-hover: #2563eb;
                --bg: #f9fafb;
                --surface: #ffffff;
                --text: #1f2937;
                --text-light: #6b7280;
                --border: #e5e7eb;
                --user-msg: #3b82f6;
                --ai-msg: #f3f4f6;
            }
            body { 
                font-family: 'Inter', sans-serif; 
                background-color: var(--bg); 
                margin: 0; 
                padding: 0; 
                display: flex; 
                justify-content: center; 
                height: 100vh;
                color: var(--text);
            }
            #app { 
                width: 100%; 
                max-width: 900px; 
                background: var(--surface); 
                display: flex; 
                flex-direction: column; 
                height: 100vh; 
                box-shadow: 0 10px 25px rgba(0,0,0,0.05); 
            }
            header { 
                background: var(--surface); 
                padding: 20px 30px; 
                text-align: center; 
                font-size: 1.5rem; 
                font-weight: 700;
                border-bottom: 1px solid var(--border);
                display: flex;
                align-items: center;
                gap: 15px;
            }
            .logo {
                width: 32px;
                height: 32px;
                background: var(--primary);
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 14px;
            }
            #messages { 
                flex-grow: 1; 
                padding: 30px; 
                overflow-y: auto; 
                display: flex; 
                flex-direction: column; 
                gap: 20px; 
                scroll-behavior: smooth;
            }
            .message-wrapper {
                display: flex;
                flex-direction: column;
                max-width: 85%;
            }
            .user-wrapper { align-self: flex-end; align-items: flex-end; }
            .ai-wrapper { align-self: flex-start; align-items: flex-start; }
            
            .message { 
                padding: 16px 20px; 
                border-radius: 12px; 
                line-height: 1.6; 
                font-size: 1rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            }
            .user-msg { 
                background: var(--user-msg); 
                color: white; 
                border-bottom-right-radius: 4px;
            }
            .ai-msg { 
                background: var(--ai-msg); 
                color: var(--text); 
                border-bottom-left-radius: 4px;
            }
            
            .source-ref { 
                font-size: 0.8em; 
                color: var(--text-light); 
                margin-top: 12px; 
                border-top: 1px solid rgba(0,0,0,0.1); 
                padding-top: 8px; 
            }
            .source-ref ul {
                margin: 5px 0 0 0;
                padding-left: 20px;
            }
            
            #input-area { 
                padding: 24px 30px; 
                background: var(--surface); 
                border-top: 1px solid var(--border); 
                display: flex; 
                gap: 12px; 
            }
            #user-input { 
                flex-grow: 1; 
                padding: 16px; 
                border: 1px solid var(--border); 
                border-radius: 8px; 
                outline: none; 
                font-size: 1rem; 
                font-family: inherit;
                transition: border-color 0.2s;
            }
            #user-input:focus {
                border-color: var(--primary);
            }
            #send-btn { 
                background: var(--primary); 
                color: white; 
                border: none; 
                padding: 0 28px; 
                border-radius: 8px; 
                cursor: pointer; 
                font-size: 1rem; 
                font-weight: 600; 
                transition: background 0.2s;
            }
            #send-btn:hover { background: var(--primary-hover); }
            #send-btn:disabled { background: var(--text-light); cursor: not-allowed; }
            
            /* Loading dots animation */
            .typing-indicator {
                display: flex;
                gap: 5px;
                padding: 16px 20px;
                background: var(--ai-msg);
                border-radius: 12px;
                border-bottom-left-radius: 4px;
                width: max-content;
            }
            .dot {
                width: 8px;
                height: 8px;
                background: var(--text-light);
                border-radius: 50%;
                animation: bounce 1.4s infinite ease-in-out both;
            }
            .dot:nth-child(1) { animation-delay: -0.32s; }
            .dot:nth-child(2) { animation-delay: -0.16s; }
            @keyframes bounce {
                0%, 80%, 100% { transform: scale(0); }
                40% { transform: scale(1); }
            }
        </style>
    </head>
    <body>
        <div id="app">
            <header>
                <div class="logo">C</div>
                Cerebro RAG
            </header>
            <div id="messages">
                <div class="message-wrapper ai-wrapper">
                    <div class="message ai-msg">Hello! I'm Cerebro. Ask me anything about your documents!</div>
                </div>
            </div>
            <div id="input-area">
                <input type="text" id="user-input" placeholder="Ask a question..." autocomplete="off">
                <button id="send-btn">Send</button>
            </div>
        </div>
        
        <script>
            const input = document.getElementById('user-input');
            const sendBtn = document.getElementById('send-btn');
            const messagesContainer = document.getElementById('messages');
            
            input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') sendMessage();
            });
            sendBtn.addEventListener('click', sendMessage);
            
            async function sendMessage() {
                const text = input.value.trim();
                if (!text) return;
                
                // Add user message
                appendMessage(text, 'user');
                input.value = '';
                toggleInput(false);
                
                // Add loading indicator
                const loadingId = addLoadingIndicator();
                
                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ query: text })
                    });
                    const data = await response.json();
                    
                    removeElement(loadingId);
                    
                    if (data.error) {
                        appendMessageHtml('Error: ' + data.error, 'ai');
                    } else {
                        let html = formatResponse(data.answer);
                        if (data.sources && data.sources.length > 0) {
                            html += '<div class="source-ref"><strong>Sources:</strong><ul>';
                            // Deduplicate sources visually
                            const uniqueSources = [...new Set(data.sources.map(s => s.doc_id))];
                            uniqueSources.forEach(s => {
                                html += "<li>" + s.split('/').pop() + "</li>";
                            });
                            html += '</ul></div>';
                        }
                        appendMessageHtml(html, 'ai');
                    }
                } catch (err) {
                    removeElement(loadingId);
                    appendMessage('Error communicating with server.', 'ai');
                } finally {
                    toggleInput(true);
                }
            }
            
            function toggleInput(enabled) {
                input.disabled = !enabled;
                sendBtn.disabled = !enabled;
                if(enabled) input.focus();
            }
            
            function appendMessage(text, sender) {
                const wrapper = document.createElement('div');
                wrapper.className = "message-wrapper " + sender + "-wrapper";
                
                const div = document.createElement('div');
                div.className = "message " + sender + "-msg";
                div.textContent = text;
                
                wrapper.appendChild(div);
                messagesContainer.appendChild(wrapper);
                scrollToBottom();
            }
            
            function appendMessageHtml(htmlStr, sender) {
                const wrapper = document.createElement('div');
                wrapper.className = "message-wrapper " + sender + "-wrapper";
                
                const div = document.createElement('div');
                div.className = "message " + sender + "-msg";
                div.innerHTML = htmlStr;
                
                wrapper.appendChild(div);
                messagesContainer.appendChild(wrapper);
                scrollToBottom();
            }
            
            function addLoadingIndicator() {
                const id = 'loading-' + Date.now();
                const wrapper = document.createElement('div');
                wrapper.id = id;
                wrapper.className = 'message-wrapper ai-wrapper';
                
                wrapper.innerHTML = `
                    <div class="typing-indicator">
                        <div class="dot"></div>
                        <div class="dot"></div>
                        <div class="dot"></div>
                    </div>
                `;
                messagesContainer.appendChild(wrapper);
                scrollToBottom();
                return id;
            }
            
            function removeElement(id) {
                const el = document.getElementById(id);
                if (el) el.remove();
            }
            
            function scrollToBottom() {
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            
            function formatResponse(text) {
                // replace newlines with <br>
                var result = '';
                var parts = text.split('\\n');
                result = parts.join('<br>');
                // replace **bold** with <strong>bold</strong>
                var out = '';
                var i = 0;
                while (i < result.length) {
                    if (i + 1 < result.length && result[i] === '*' && result[i+1] === '*') {
                        var end = result.indexOf('**', i + 2);
                        if (end !== -1) {
                            out += '<strong>' + result.substring(i + 2, end) + '</strong>';
                            i = end + 2;
                        } else {
                            out += result[i];
                            i++;
                        }
                    } else {
                        out += result[i];
                        i++;
                    }
                }
                return out;
            }
        </script>
    </body>
    </html>
    """)
end

function setup_routes()
    route("/", chat_ui)

    route("/api/chat", method=POST) do
        req_data = jsonpayload()
        if !haskey(req_data, "query")
            return json(Dict("error" => "query is required"))
        end

        query = req_data["query"]
        parents, children, index = GLOBAL_DB[]

        if isempty(parents)
            return json(Dict("answer" => "The database is empty. Please add documents to the folder and restart.", "sources" => []))
        end

        println("=> Retrieving context for query: ", query)
        top_parents = retrieve_context(query, parents, children, index)

        println("=> Generating answer...")
        answer = generate_answer(query, top_parents)

        sources = [Dict("id" => p.id, "doc_id" => p.doc_id) for p in top_parents]
        println("=> Sending response")

        return json(Dict("answer" => answer, "sources" => sources))
    end
end

function start_server(; port=8000, host="127.0.0.1", folder_path="data/documents")
    println("=== Starting Cerebro ===")
    init_db(folder_path)
    setup_routes()

    # Configure Genie
    Genie.config.run_as_server = true
    Genie.config.server_port = port
    Genie.config.server_host = host

    println("=> Server running at http://\$host:\$port")
    Genie.up()
end

end # module
