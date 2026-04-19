(function () {
  const socket = io({ transports: ['websocket'] });
  const snapshotEl = document.getElementById('snapshot');
  const noFeedEl = document.getElementById('no-feed');
  const eventList = document.getElementById('event-list');
  const statusEl = document.getElementById('status-indicator');
  const lastTsEl = document.getElementById('last-snapshot-ts');

  socket.on('connect', () => {
    statusEl.textContent = '● Connected';
    statusEl.classList.add('connected');
  });

  socket.on('disconnect', () => {
    statusEl.textContent = '● Disconnected';
    statusEl.classList.remove('connected');
  });

  socket.on('snapshot', ({ data }) => {
    if (!data) return;
    snapshotEl.src = 'data:image/jpeg;base64,' + data;
    snapshotEl.classList.add('loaded');
    noFeedEl.style.display = 'none';
    lastTsEl.textContent = 'Last update: ' + new Date().toLocaleTimeString();
  });

  socket.on('new_event', (ev) => {
    prependEvent(ev);
  });

  // Load existing events on page load
  fetch('/api/events')
    .then(r => r.json())
    .then(events => events.forEach(ev => appendEvent(ev)))
    .catch(() => {});

  fetch('/api/latest-snapshot')
    .then(r => r.json())
    .then(({ snapshot }) => {
      if (snapshot) {
        snapshotEl.src = 'data:image/jpeg;base64,' + snapshot;
        snapshotEl.classList.add('loaded');
        noFeedEl.style.display = 'none';
      }
    })
    .catch(() => {});

  function prependEvent(ev) {
    const li = buildEventItem(ev);
    eventList.prepend(li);
    // Keep list bounded
    while (eventList.children.length > 100) {
      eventList.removeChild(eventList.lastChild);
    }
  }

  function appendEvent(ev) {
    const li = buildEventItem(ev);
    eventList.appendChild(li);
  }

  function buildEventItem(ev) {
    const li = document.createElement('li');
    li.className = 'event-item';

    const name = ev.visitor_name || 'Unknown';
    const known = ev.known === true;
    const objects = (ev.objects || []).join(', ');
    const ts = ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : '';

    li.innerHTML = `
      <div class="event-header">
        <span class="badge ${known ? 'known' : 'unknown'}">${known ? 'Known' : 'Unknown'}</span>
        <span class="event-name">${escHtml(name)}</span>
        <span class="event-time">${escHtml(ts)}</span>
      </div>
      ${objects ? `<div class="event-objects">Detected: ${escHtml(objects)}</div>` : ''}
    `;
    return li;
  }

  function escHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
})();
