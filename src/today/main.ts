import { deriveDashboard, type DashboardModel } from './derive';
import { readTodaySources } from './data';
import type { Exercise, HabitStatus, SlateBlock, TrendWeek } from './types';

const dashboard = document.querySelector<HTMLElement>('#dashboard');
const dateNode = document.querySelector<HTMLElement>('#today-date');
const syncNode = document.querySelector<HTMLElement>('#sync-state');
const refreshButton = document.querySelector<HTMLButtonElement>('#refresh-button');

function escapeHtml(value: unknown): string {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function setHtml(id: string, html: string): void {
  const node = document.querySelector<HTMLElement>(`#${id}`);
  if (node) node.innerHTML = html;
}

function formatTime(minutes: number): string {
  const date = new Date(2000, 0, 1, Math.floor(minutes / 60), minutes % 60);
  return new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' }).format(date);
}

function formatDuration(minutes: number): string {
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  const remainder = minutes % 60;
  return remainder ? `${hours}h ${remainder}m` : `${hours}h`;
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(Math.round(value * 10) / 10);
}

function panelHeader(kicker: string, title: string, href: string, label: string): string {
  return `
    <header class="panel-header">
      <div><span class="eyebrow">${escapeHtml(kicker)}</span><h2>${escapeHtml(title)}</h2></div>
      <a class="source-link" href="${href}" aria-label="Open ${escapeHtml(label)}">${escapeHtml(label)} <span aria-hidden="true">↗</span></a>
    </header>`;
}

function emptyState(app: string, href: string, message: string): string {
  return `
    <div class="empty-state">
      <span class="empty-mark" aria-hidden="true">○</span>
      <p>${escapeHtml(message)}</p>
      <a href="${href}">Open ${escapeHtml(app)} to connect this device <span aria-hidden="true">↗</span></a>
    </div>`;
}

function hero(model: DashboardModel): void {
  const title = document.querySelector<HTMLElement>('#hero-title');
  const detail = document.querySelector<HTMLElement>('#hero-detail');
  const kicker = document.querySelector<HTMLElement>('#hero-kicker');
  if (!title || !detail || !kicker) return;

  if (model.currentBlock) {
    kicker.textContent = `Now · until ${formatTime(model.currentBlock.startMin + model.currentBlock.durationMin)}`;
    title.textContent = model.currentBlock.title;
    detail.textContent = model.priority && model.priority.task.title !== model.currentBlock.title
      ? `Then protect the main unfinished task: ${model.priority.task.title}.`
      : 'This is the active block on your Slate.';
  } else if (model.priority && /Overdue|Due today/.test(model.priority.reason)) {
    kicker.textContent = model.priority.reason;
    title.textContent = model.priority.task.title;
    detail.textContent = `This ranks first across ${model.dueTasks.length} due or overdue unfinished task${model.dueTasks.length === 1 ? '' : 's'}.`;
  } else if (model.nextBlock) {
    kicker.textContent = `Next · ${formatTime(model.nextBlock.startMin)}`;
    title.textContent = model.nextBlock.title;
    detail.textContent = model.priority ? `Before then, the clearest open loop is ${model.priority.task.title}.` : 'Your next scheduled block is already protected.';
  } else if (model.workout?.exercises.length) {
    kicker.textContent = 'Training plan';
    title.textContent = model.workout.blocks[0]?.label ?? 'Today’s workout';
    detail.textContent = `${model.workout.exercises.length} movements${model.workout.courtSports.length ? `, including ${model.workout.courtSports.map((sport) => sport.name).join(' and ')}` : ''}.`;
  } else if (model.dueHabits.length) {
    kicker.textContent = 'Open obligation';
    title.textContent = model.dueHabits[0].habit.name;
    detail.textContent = model.dueHabits[0].label;
  } else {
    kicker.textContent = 'Today';
    title.textContent = model.connectedCount ? 'Nothing urgent is asking for attention.' : 'Connect the four apps on this device.';
    detail.textContent = model.connectedCount ? 'Use the source cards below for the rest of the day.' : 'Open each source app once, then refresh this view.';
  }

  setHtml('hero-meta', `
    <div><strong>${model.blocks.length}</strong><span>blocks</span></div>
    <div><strong>${model.dueTasks.length}</strong><span>tasks due</span></div>
    <div><strong>${model.dueHabits.length}</strong><span>habits open</span></div>
  `);
}

function renderPriority(model: DashboardModel): void {
  let content: string;
  if (!model.sources.slate.connected) {
    content = emptyState('Slate', '/slate/', 'Slate has not left a readable task list on this device yet.');
  } else if (!model.priority) {
    content = '<div class="resolved-state"><span aria-hidden="true">✓</span><strong>No unfinished tasks</strong><p>Slate is clear.</p></div>';
  } else {
    const { task, section, reason } = model.priority;
    content = `
      <div class="priority-task">
        <span class="reason-chip${reason.startsWith('Overdue') ? ' critical' : ''}">${escapeHtml(reason)}</span>
        <h3>${escapeHtml(task.title)}</h3>
        <p>${escapeHtml(section)}${task.notes ? ` · ${escapeHtml(task.notes)}` : ''}</p>
      </div>
      <div class="ranking-note"><span>Ranked by</span><strong>overdue → due today → scheduled → list order</strong></div>`;
  }
  setHtml('priority-card', `${panelHeader('Priority', 'Most important unfinished', '/slate/', 'Slate')}${content}`);
}

function blockRow(block: SlateBlock, nowMinutes: number): string {
  const active = block.startMin <= nowMinutes && block.startMin + block.durationMin > nowMinutes;
  const past = block.startMin + block.durationMin <= nowMinutes;
  return `
    <li class="timeline-row${active ? ' active' : ''}${past ? ' past' : ''}">
      <time>${formatTime(block.startMin)}</time>
      <span class="timeline-rule" style="--block-color:${escapeHtml(block.color)}"></span>
      <div><strong>${escapeHtml(block.title)}</strong><span>${formatDuration(block.durationMin)}${active ? ' · now' : ''}</span></div>
    </li>`;
}

function renderSchedule(model: DashboardModel): void {
  const minute = model.now.getHours() * 60 + model.now.getMinutes();
  const content = !model.sources.slate.connected
    ? emptyState('Slate', '/slate/', 'Today’s schedule will appear after Slate is opened on this device.')
    : model.blocks.length
      ? `<ol class="timeline">${model.blocks.map((block) => blockRow(block, minute)).join('')}</ol>`
      : '<div class="resolved-state quiet"><strong>No time blocks today</strong><p>The day is unscheduled in Slate.</p></div>';
  setHtml('schedule-card', `${panelHeader('Plan', 'What today asks of you', '/slate/', 'Slate')}${content}`);
}

function habitRow(status: HabitStatus, kind: 'overdue' | 'due' | 'flexible'): string {
  const meta = kind === 'overdue'
    ? `${status.previousMisses} missed scheduled day${status.previousMisses === 1 ? '' : 's'} in the last 14`
    : kind === 'flexible'
      ? `${status.label} · ${status.habit.period} window`
      : status.label;
  return `
    <li class="habit-row">
      <div><strong>${escapeHtml(status.habit.name)}</strong><span>${escapeHtml(meta)}</span></div>
      <div class="mini-progress" role="progressbar" aria-label="${escapeHtml(status.habit.name)} progress" aria-valuenow="${Math.round(status.ratio * 100)}" aria-valuemin="0" aria-valuemax="100">
        <span style="width:${Math.round(status.ratio * 100)}%"></span>
      </div>
    </li>`;
}

function habitGroup(title: string, statuses: HabitStatus[], kind: 'overdue' | 'due' | 'flexible'): string {
  if (!statuses.length) return '';
  return `<section class="habit-group"><h3>${escapeHtml(title)} <span>${statuses.length}</span></h3><ul>${statuses.map((status) => habitRow(status, kind)).join('')}</ul></section>`;
}

function renderHabits(model: DashboardModel): void {
  let content: string;
  if (!model.sources.daymark.connected) {
    content = emptyState('Daymark', '/daymark/', 'Habit obligations will appear after Daymark is opened on this device.');
  } else {
    const overdueIds = new Set(model.overdueHabits.map((status) => status.habit.id));
    const dueOnly = model.dueHabits.filter((status) => !overdueIds.has(status.habit.id));
    content = [
      habitGroup('Truly overdue', model.overdueHabits, 'overdue'),
      habitGroup('Due today', dueOnly, 'due'),
      habitGroup('Flexible', model.flexibleHabits, 'flexible'),
    ].join('') || '<div class="resolved-state"><span aria-hidden="true">✓</span><strong>Habits handled</strong><p>No open habit obligations remain today.</p></div>';
  }
  setHtml('habits-card', `${panelHeader('Consistency', 'Flexible vs. overdue', '/daymark/', 'Daymark')}${content}`);
}

function nutritionMetric(label: string, remaining: number, consumed: number, target: number, unit: string): string {
  const over = Math.max(0, consumed - target);
  const percent = target ? Math.min(100, Math.round(consumed / target * 100)) : 0;
  return `
    <div class="nutrition-metric">
      <div class="metric-top"><span>${escapeHtml(label)}</span><small>${formatNumber(consumed)} / ${formatNumber(target)} ${unit}</small></div>
      <strong>${formatNumber(over || remaining)}<small>${unit}</small></strong>
      <span class="metric-caption">${over ? 'over target' : 'remaining'}</span>
      <div class="macro-track" role="progressbar" aria-label="${escapeHtml(label)} consumed" aria-valuenow="${percent}" aria-valuemin="0" aria-valuemax="100"><span style="width:${percent}%"></span></div>
    </div>`;
}

function renderNutrition(model: DashboardModel): void {
  const content = !model.sources.fare.connected || !model.nutrition
    ? emptyState('Fare', '/fare/', 'Calorie and protein remainder will appear after Fare is opened on this device.')
    : `
      <div class="nutrition-grid">
        ${nutritionMetric('Calories', model.nutrition.remainingCalories, model.nutrition.consumedCalories, model.nutrition.targetCalories, 'kcal')}
        ${nutritionMetric('Protein', model.nutrition.remainingProtein, model.nutrition.consumedProtein, model.nutrition.targetProtein, 'g')}
      </div>
      <p class="card-footnote">Based on ${model.nutrition.entries} item${model.nutrition.entries === 1 ? '' : 's'} logged today. Targets and history stay owned by Fare.</p>`;
  setHtml('nutrition-card', `${panelHeader('Fuel', 'What remains', '/fare/', 'Fare')}${content}`);
}

function exerciseTarget(exercise: Exercise): string {
  if (exercise.target?.minutes) return `${exercise.target.minutes} min`;
  if (exercise.target?.sets) return `${exercise.target.sets} sets`;
  return exercise.kind;
}

function renderWorkout(model: DashboardModel): void {
  let content: string;
  if (!model.sources.gym.connected || !model.workout) {
    content = emptyState('Gym', '/gym/', 'The scheduled workout will appear after Gym is opened on this device.');
  } else if (!model.workout.exercises.length) {
    content = '<div class="resolved-state quiet"><strong>No stored workout for today</strong><p>Open Gym to load or adjust the weekly program.</p></div>';
  } else {
    const completed = new Set(model.workout.log?.completed ?? []);
    const skipped = new Set(model.workout.log?.skipped ?? []);
    content = `
      ${model.workout.courtSports.length ? `<div class="court-callout"><span>Today</span><strong>${model.workout.courtSports.map((sport) => escapeHtml(sport.name.replace(/\s+minutes/i, ' min'))).join(' + ')}</strong></div>` : '<div class="court-callout quiet"><span>Court</span><strong>No basketball or badminton in today’s Gym plan</strong></div>'}
      <div class="workout-progress"><strong>${model.workout.completed}<span> / ${model.workout.exercises.length}</span></strong><p>movements complete${model.workout.skipped ? ` · ${model.workout.skipped} skipped` : ''}</p></div>
      <div class="workout-blocks">
        ${model.workout.blocks.map((block) => `
          <section><h3>${escapeHtml(block.label)}</h3><ul>
            ${block.exercises.map((exercise) => `<li class="${completed.has(exercise.id) ? 'done' : skipped.has(exercise.id) ? 'skipped' : ''}"><span>${completed.has(exercise.id) ? '✓' : skipped.has(exercise.id) ? '–' : '○'}</span><strong>${escapeHtml(exercise.name)}</strong><small>${escapeHtml(exerciseTarget(exercise))}</small></li>`).join('')}
          </ul></section>`).join('')}
      </div>`;
  }
  setHtml('workout-card', `${panelHeader('Training', 'Scheduled workout', '/gym/', 'Gym')}${content}`);
}

function metricDelta(weeks: TrendWeek[], metric: keyof Pick<TrendWeek, 'nutrition' | 'training' | 'habits'>): string {
  const previous = weeks[2][metric];
  const current = weeks[3][metric];
  if (previous === null || current === null) return '—';
  const delta = Math.round((current - previous) * 100);
  return `${delta > 0 ? '+' : ''}${delta} pts`;
}

function trendCells(weeks: TrendWeek[], metric: keyof Pick<TrendWeek, 'nutrition' | 'training' | 'habits'>): string {
  return weeks.map((week) => {
    const value = week[metric];
    const height = value === null ? 4 : Math.max(4, Math.round(value * 100));
    return `<span class="trend-bar${value === null ? ' missing' : ''}" style="height:${height}%" title="${escapeHtml(week.label)}: ${value === null ? 'No data' : `${Math.round(value * 100)}%`}"></span>`;
  }).join('');
}

function trendRow(label: string, metric: keyof Pick<TrendWeek, 'nutrition' | 'training' | 'habits'>, weeks: TrendWeek[]): string {
  return `
    <div class="trend-row">
      <div class="trend-label"><strong>${escapeHtml(label)}</strong><span>${metricDelta(weeks, metric)}</span></div>
      <div class="trend-bars" aria-label="Four-week ${escapeHtml(label.toLowerCase())} trend">${trendCells(weeks, metric)}</div>
    </div>`;
}

function renderTrends(model: DashboardModel): void {
  const hasData = model.trends.some((week) => week.nutrition !== null || week.training !== null || week.habits !== null);
  const content = hasData
    ? `
      <div class="trend-summary"><p>${escapeHtml(model.trendNarrative)}</p><span>Latest 7 days vs. previous 7</span></div>
      <div class="trend-list">
        ${trendRow('Nutrition', 'nutrition', model.trends)}
        ${trendRow('Training', 'training', model.trends)}
        ${trendRow('Habits', 'habits', model.trends)}
      </div>
      <div class="trend-axis">${model.trends.map((week) => `<span>${escapeHtml(week.label)}</span>`).join('')}</div>`
    : '<div class="empty-state"><span class="empty-mark" aria-hidden="true">↗</span><p>Consistency trends will form from the existing app histories.</p></div>';
  setHtml('trend-card', `${panelHeader('28 days', 'How the systems move together', '#source-strip', 'Sources')}${content}`);
}

function renderSources(model: DashboardModel): void {
  const apps = [
    ['Daymark', '/daymark/', model.sources.daymark],
    ['Slate', '/slate/', model.sources.slate],
    ['Fare', '/fare/', model.sources.fare],
    ['Gym', '/gym/', model.sources.gym],
  ] as const;
  setHtml('source-strip', `
    <div class="source-intro"><span class="eyebrow">On this device</span><strong>${model.connectedCount} of 4 sources connected</strong></div>
    <nav aria-label="Source apps">
      ${apps.map(([name, href, source]) => `<a href="${href}" class="source-pill ${source.connected ? 'connected' : ''}"><span aria-hidden="true"></span><strong>${name}</strong><small>${source.connected ? source.source : 'open to connect'}</small></a>`).join('')}
    </nav>
    <p>Today reads existing on-device data and never creates tasks, habits, meals, or workouts.</p>`);
}

function render(model: DashboardModel): void {
  if (dateNode) {
    dateNode.textContent = new Intl.DateTimeFormat(undefined, { weekday: 'long', month: 'long', day: 'numeric' }).format(model.now);
  }
  hero(model);
  renderPriority(model);
  renderSchedule(model);
  renderHabits(model);
  renderNutrition(model);
  renderWorkout(model);
  renderTrends(model);
  renderSources(model);
  dashboard?.setAttribute('aria-busy', 'false');
}

let loading = false;
async function refresh(): Promise<void> {
  if (loading) return;
  loading = true;
  refreshButton?.classList.add('is-loading');
  if (syncNode) syncNode.textContent = 'Reading this device';
  try {
    const sources = await readTodaySources();
    const model = deriveDashboard(sources);
    render(model);
    if (syncNode) syncNode.textContent = `${model.connectedCount}/4 connected · just now`;
  } catch {
    if (syncNode) syncNode.textContent = 'Could not refresh';
  } finally {
    loading = false;
    refreshButton?.classList.remove('is-loading');
  }
}

refreshButton?.addEventListener('click', () => void refresh());
window.addEventListener('storage', () => void refresh());
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') void refresh();
});

void refresh();
