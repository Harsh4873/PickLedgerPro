const PAGE_URL = 'http://127.0.0.1:8000/pickledger.html?api=http://127.0.0.1:8765';
const DEBUG_BASE = 'http://127.0.0.1:9222';

async function createPage() {
  const response = await fetch(`${DEBUG_BASE}/json/new?${PAGE_URL}`, { method: 'PUT' });
  if (!response.ok) throw new Error(`Unable to create CDP page: ${response.status}`);
  return response.json();
}

class CdpClient {
  constructor(pageInfo) {
    this.pageInfo = pageInfo;
    this.ws = new WebSocket(pageInfo.webSocketDebuggerUrl);
    this.nextId = 1;
    this.pending = new Map();
    this.runtimeErrors = [];
    this.ws.addEventListener('message', (event) => {
      const message = JSON.parse(event.data);
      if (message.id) {
        const resolver = this.pending.get(message.id);
        if (resolver) {
          this.pending.delete(message.id);
          resolver(message);
        }
        return;
      }
      if (message.method === 'Runtime.exceptionThrown') {
        this.runtimeErrors.push(message.params.exceptionDetails?.text || 'Runtime exception');
      }
      if (message.method === 'Log.entryAdded') {
        const entry = message.params.entry || {};
        if (entry.level === 'error') this.runtimeErrors.push(entry.text || 'Log error');
      }
    });
  }

  async open() {
    await new Promise((resolve, reject) => {
      this.ws.addEventListener('open', resolve, { once: true });
      this.ws.addEventListener('error', reject, { once: true });
    });
    await this.send('Page.enable');
    await this.send('Runtime.enable');
    await this.send('Log.enable');
  }

  async close() {
    try {
      await fetch(`${DEBUG_BASE}/json/close/${this.pageInfo.id}`);
    } catch {}
    try {
      this.ws.close();
    } catch {}
  }

  send(method, params = {}) {
    const id = this.nextId++;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => {
      this.pending.set(id, (message) => {
        if (message.error) reject(new Error(message.error.message || method));
        else resolve(message.result || {});
      });
      setTimeout(() => {
        if (!this.pending.has(id)) return;
        this.pending.delete(id);
        reject(new Error(`CDP timeout: ${method}`));
      }, 180000);
    });
  }

  async evaluate(expression) {
    const result = await this.send('Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (result.exceptionDetails) {
      throw new Error(result.exceptionDetails.text || `Evaluation failed: ${expression}`);
    }
    return result.result ? result.result.value : undefined;
  }

  async waitFor(expression, timeout = 30000, interval = 250) {
    const start = Date.now();
    while (Date.now() - start < timeout) {
      const ok = await this.evaluate(expression).catch(() => false);
      if (ok) return true;
      await new Promise((resolve) => setTimeout(resolve, interval));
    }
    throw new Error(`Timed out waiting for: ${expression}`);
  }

  async waitForStatus(selector, okPattern, errorPattern, timeout = 180000) {
    const start = Date.now();
    while (Date.now() - start < timeout) {
      const text = await this.evaluate(`(document.querySelector(${JSON.stringify(selector)})?.textContent || '').trim()`);
      if (text && new RegExp(errorPattern, 'i').test(text)) {
        throw new Error(`Status error at ${selector}: ${text}`);
      }
      if (text && new RegExp(okPattern, 'i').test(text)) {
        return text;
      }
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
    throw new Error(`Timed out waiting for status ${selector}`);
  }
}

async function main() {
  const summary = {
    headerHasIplRecord: null,
    iplWinnerAdded: false,
    iplFantasyAdded: false,
    iplPicksVisibleInLog: false,
    iplPendingBeforeManualGrade: false,
    manualGradeWorked: false,
    iplApiPicksRequests: 0,
    mlbGenericFlowRendered: false,
    mlbGenericAddWorked: false,
    nbaGenericFlowRendered: false,
    nbaGenericAddWorked: false,
    runtimeErrors: [],
  };

  const pageInfo = await createPage();
  const cdp = new CdpClient(pageInfo);
  try {
    await cdp.open();
    await cdp.waitFor('document.readyState === "complete" && typeof runModel === "function"', 30000);
    await cdp.evaluate(`(() => {
      window.__fetchUrls = [];
      const originalFetch = window.fetch.bind(window);
      window.fetch = (...args) => {
        try { window.__fetchUrls.push(String(args[0])); } catch {}
        return originalFetch(...args);
      };
      return true;
    })()`);

    summary.headerHasIplRecord = await cdp.evaluate(
      `Boolean(document.getElementById('ipl-record')) || /IPL:\\s*\\d+-\\d+/i.test(document.querySelector('header')?.innerText || '')`
    );

    await cdp.evaluate(`switchTab('models'); true;`);
    await cdp.evaluate(`runModel('ipl'); true;`);
    await cdp.waitFor(`document.querySelector('#ipl-prediction-root .ipl-winner-card') !== null`, 180000);
    await cdp.waitForStatus('#status-ipl', 'Done', 'Error', 180000);

    const genericButtonsHidden = await cdp.evaluate(`(() => {
      const hidden = (id) => {
        const el = document.getElementById(id);
        return !el || getComputedStyle(el).display === 'none';
      };
      return hidden('btn-add-selected') && hidden('btn-add-picks');
    })()`);
    if (!genericButtonsHidden) {
      throw new Error('Generic add buttons are visible during IPL view');
    }

    await cdp.evaluate(`addIplWinnerPick(); true;`);
    await cdp.waitFor(`(() => {
      const btn = document.querySelector('.winner-callout .ipl-add-check');
      return Boolean(btn && btn.disabled && /Added/i.test(btn.textContent || ''));
    })()`, 15000);
    summary.iplWinnerAdded = true;

    await cdp.evaluate(`addIplFantasyPick(0); true;`);
    await cdp.waitFor(`(() => {
      const btn = document.querySelector('.ipl-fantasy-table tbody .add-col .ipl-add-check');
      return Boolean(btn && btn.disabled && /Added/i.test(btn.textContent || ''));
    })()`, 15000);
    summary.iplFantasyAdded = true;

    summary.iplApiPicksRequests = await cdp.evaluate(
      `window.__fetchUrls.filter((url) => /\\/api\\/picks/i.test(url)).length`
    );

    await cdp.evaluate(`switchTab('home'); render(); true;`);
    await cdp.waitFor(`document.querySelectorAll('#pick-feed .pick-bubble').length >= 2`, 15000);
    summary.iplPicksVisibleInLog = await cdp.evaluate(`(() => {
      const texts = Array.from(document.querySelectorAll('#pick-feed .pick-bubble-pick')).map((el) => el.textContent || '');
      return texts.some((text) => text.startsWith('WINNER:')) && texts.some((text) => text.startsWith('FANTASY:'));
    })()`);
    summary.iplPendingBeforeManualGrade = await cdp.evaluate(
      `Array.from(document.querySelectorAll('#pick-feed .result-select')).slice(0, 2).every((el) => el.value === 'pending')`
    );

    await cdp.evaluate(`(() => {
      const select = document.querySelector('#pick-feed .result-select');
      select.value = 'win';
      select.dispatchEvent(new Event('change', { bubbles: true }));
      return true;
    })()`);
    await cdp.waitFor(`Object.values(getResults()).includes('win')`, 15000);
    summary.manualGradeWorked = true;

    await cdp.evaluate(`switchTab('models'); true;`);
    await cdp.evaluate(`(() => {
      const select = document.getElementById('scores24-league-select');
      select.value = 'mlb';
      return runSelectedScores24League();
    })()`);
    await cdp.waitForStatus('#status-scores24', 'Synced|Loaded|Using cached', 'failed|Error', 120000);
    summary.mlbGenericFlowRendered = await cdp.evaluate(`(() => {
      const hasRows = document.querySelectorAll('#model-results-body tr').length > 0;
      const selected = document.getElementById('btn-add-selected');
      const all = document.getElementById('btn-add-picks');
      return hasRows && getComputedStyle(selected).display !== 'none' && getComputedStyle(all).display !== 'none';
    })()`);
    await cdp.evaluate(`(() => {
      document.querySelectorAll('.model-pick-cb').forEach((cb, index) => { cb.checked = index === 0; });
      updateModelSelectAll();
      addSelectedPicksToLedger();
      return true;
    })()`);
    await cdp.waitFor(`getPicks().some((pick) => String(pick.sport || '').toUpperCase() === 'MLB')`, 15000);
    summary.mlbGenericAddWorked = true;

    await cdp.evaluate(`switchTab('models'); true;`);
    await cdp.evaluate(`(() => {
      const select = document.getElementById('scores24-league-select');
      select.value = 'nba';
      return runSelectedScores24League();
    })()`);
    await cdp.waitForStatus('#status-scores24', 'Synced|Loaded|Using cached', 'failed|Error', 120000);
    summary.nbaGenericFlowRendered = await cdp.evaluate(`(() => {
      const hasRows = document.querySelectorAll('#model-results-body tr').length > 0;
      const selected = document.getElementById('btn-add-selected');
      const all = document.getElementById('btn-add-picks');
      return hasRows && getComputedStyle(selected).display !== 'none' && getComputedStyle(all).display !== 'none';
    })()`);
    await cdp.evaluate(`(() => {
      document.querySelectorAll('.model-pick-cb').forEach((cb, index) => { cb.checked = index === 0; });
      updateModelSelectAll();
      addSelectedPicksToLedger();
      return true;
    })()`);
    await cdp.waitFor(`getPicks().some((pick) => String(pick.sport || '').toUpperCase() === 'NBA')`, 15000);
    summary.nbaGenericAddWorked = true;

    summary.runtimeErrors = cdp.runtimeErrors;
    console.log(JSON.stringify(summary, null, 2));
  } catch (error) {
    summary.runtimeErrors = cdp.runtimeErrors;
    console.error(JSON.stringify({ ...summary, error: String(error && error.message ? error.message : error) }, null, 2));
    process.exitCode = 1;
  } finally {
    await cdp.close();
  }
}

main();
