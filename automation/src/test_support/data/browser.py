"""Offline chat DOM and deterministic response states for Page Object regressions."""

CHAT_HTML = """
        <div id="chat-history"></div>
        <textarea placeholder="Send a message"></textarea>
        <button aria-label="Send prompt message to workspace">Send</button>
        <script>
          window.reply = (text, sources = false) => {
            const reply = document.createElement('div');
            reply.className = 'group';
            const content = document.createElement('div');
            content.className = 'break-words';
            content.textContent = text;
            reply.appendChild(content);
            const edit = document.createElement('button');
            edit.setAttribute('aria-label', 'Edit Edit response');
            reply.appendChild(edit);
            if (sources) {
              const button = document.createElement('button');
              button.textContent = 'Sources';
              reply.appendChild(button);
            }
            document.querySelector('#chat-history').appendChild(reply);
          };
          document.querySelector('button').onclick = () => {
            setTimeout(() => window.reply('New final answer'), 200);
          };
        </script>
    """

EMPTY_THEN_READY_REPLY = """() => {
    window.reply('');
    setTimeout(() => document.querySelector('.break-words').textContent = 'Ready', 200);
}"""

COMPLETED_REPLY_WITH_DISABLED_SEND = """() => {
    window.reply('Complete');
    document.querySelector('button[aria-label="Send prompt message to workspace"]').disabled = true;
}"""
