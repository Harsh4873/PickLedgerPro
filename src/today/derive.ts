import type {
  DateKey,
  DaymarkState,
  Exercise,
  FareState,
  Habit,
  HabitEntry,
  HabitStatus,
  SlateState,
  SlateTask,
  TodaySources,
  TrendWeek,
  WorkoutLog,
} from './types';

export interface NutritionSummary {
  consumedCalories: number;
  consumedProtein: number;
  targetCalories: number;
  targetProtein: number;
  remainingCalories: number;
  remainingProtein: number;
  entries: number;
}

export interface WorkoutBlock {
  label: string;
  exercises: Exercise[];
}

export interface WorkoutSummary {
  exercises: Exercise[];
  blocks: WorkoutBlock[];
  log: WorkoutLog | null;
  completed: number;
  skipped: number;
  courtSports: Exercise[];
}

export interface PriorityTask {
  task: SlateTask;
  section: string;
  reason: string;
}

export interface AdviceInsight {
  id: string;
  title: string;
  detail: string;
  evidence: string;
  source: 'Daymark' | 'Slate' | 'Fare' | 'Gym' | 'Today';
  href: string;
  priority: number;
  tone: 'act' | 'prepare' | 'protect' | 'positive';
}

export interface DayPressure {
  score: number;
  label: 'Light' | 'Manageable' | 'Loaded' | 'Tight';
  confidence: 'Low' | 'Medium' | 'High';
  basis: 'load-only';
  factors: string[];
}

export interface DashboardModel {
  now: Date;
  todayKey: DateKey;
  priority: PriorityTask | null;
  dueTasks: SlateTask[];
  openTasks: number;
  dueHabits: HabitStatus[];
  flexibleHabits: HabitStatus[];
  overdueHabits: HabitStatus[];
  nutrition: NutritionSummary | null;
  workout: WorkoutSummary | null;
  trends: TrendWeek[];
  trendNarrative: string;
  advice: AdviceInsight[];
  pressure: DayPressure;
  connectedCount: number;
  sources: TodaySources;
}

export function toDateKey(date: Date): DateKey {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

export function fromDateKey(key: DateKey): Date {
  const [year, month, day] = key.split('-').map(Number);
  return new Date(year, month - 1, day, 12);
}

export function addDays(date: Date, amount: number): Date {
  const next = new Date(date.getFullYear(), date.getMonth(), date.getDate(), 12);
  next.setDate(next.getDate() + amount);
  return next;
}

function daysBetween(start: Date, end: Date): Date[] {
  const days: Date[] = [];
  for (let cursor = new Date(start); cursor <= end; cursor = addDays(cursor, 1)) days.push(cursor);
  return days;
}

function startOfWeek(date: Date, weekStartsOn: 0 | 1): Date {
  const offset = (date.getDay() - weekStartsOn + 7) % 7;
  return addDays(date, -offset);
}

function endOfPeriod(period: Habit['period'], date: Date, weekStartsOn: 0 | 1): Date {
  if (period === 'week') return addDays(startOfWeek(date, weekStartsOn), 6);
  if (period === 'month') return new Date(date.getFullYear(), date.getMonth() + 1, 0, 12);
  return date;
}

function startOfPeriod(period: Habit['period'], date: Date, weekStartsOn: 0 | 1): Date {
  if (period === 'week') return startOfWeek(date, weekStartsOn);
  if (period === 'month') return new Date(date.getFullYear(), date.getMonth(), 1, 12);
  return date;
}

function isHabitActive(habit: Habit, date: Date): boolean {
  const key = toDateKey(date);
  if (key < habit.startDate) return false;
  if (habit.pauses?.some((pause) => key >= pause.start && (!pause.end || key < pause.end))) return false;
  if (habit.archivedAt && !habit.pauses?.length && key >= habit.archivedAt.slice(0, 10)) return false;
  return true;
}

function isHabitScheduled(habit: Habit, date: Date): boolean {
  if (!isHabitActive(habit, date)) return false;
  if (habit.period !== 'day') return true;
  if (habit.schedule.type === 'everyday') return true;
  if (habit.schedule.type === 'selectedDays') return Boolean(habit.schedule.days?.includes(date.getDay()));
  const elapsed = Math.round((fromDateKey(toDateKey(date)).getTime() - fromDateKey(habit.startDate).getTime()) / 86_400_000);
  const interval = Math.max(1, habit.schedule.every ?? 1) * (habit.schedule.unit === 'week' ? 7 : 1);
  return elapsed >= 0 && elapsed % interval === 0;
}

function entryFor(state: DaymarkState, habit: Habit, date: Date): HabitEntry | undefined {
  return state.entries[toDateKey(date)]?.[habit.id];
}

function periodStatus(state: DaymarkState, habit: Habit, date: Date): HabitStatus {
  const start = startOfPeriod(habit.period, date, state.profile.weekStartsOn);
  const end = endOfPeriod(habit.period, date, state.profile.weekStartsOn);
  const entries = daysBetween(start, end)
    .filter((day) => isHabitActive(habit, day))
    .map((day) => entryFor(state, habit, day))
    .filter((entry): entry is HabitEntry => Boolean(entry && !entry.skipped && entry.hasValue !== false));
  const value = entries.reduce((sum, entry) => sum + Math.max(0, Number(entry.value) || 0), 0);
  const hasEntry = entries.length > 0;
  const complete = habit.direction === 'atMost'
    ? hasEntry && value <= habit.target && (habit.period !== 'day' || toDateKey(end) < toDateKey(new Date()))
    : value >= habit.target;
  const ratio = habit.direction === 'atMost'
    ? hasEntry ? Math.min(1, habit.target / Math.max(value, 1)) : 0
    : Math.min(1, value / Math.max(habit.target, 0.0001));
  const remaining = Math.max(0, habit.target - value);
  const label = complete
    ? 'Complete'
    : `${formatNumber(remaining)}${habit.unit ? ` ${habit.unit}` : ''} left`;
  return { habit, value, target: habit.target, ratio, complete, label, previousMisses: 0 };
}

function habitHandled(state: DaymarkState, habit: Habit, date: Date): boolean {
  const entry = entryFor(state, habit, date);
  if (entry?.skipped) return true;
  return periodStatus(state, habit, date).complete;
}

function previousMisses(state: DaymarkState, habit: Habit, today: Date): number {
  if (habit.period !== 'day') return 0;
  let misses = 0;
  for (let offset = 1; offset <= 14; offset += 1) {
    const date = addDays(today, -offset);
    if (date < fromDateKey(habit.startDate)) break;
    if (isHabitScheduled(habit, date) && !habitHandled(state, habit, date)) misses += 1;
  }
  return misses;
}

function formatNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1).replace(/\.0$/, '');
}

function deriveHabits(state: DaymarkState | null, today: Date) {
  if (!state) return { due: [], flexible: [], overdue: [] };
  const active = state.habits.filter((habit) => isHabitActive(habit, today));
  const due = active
    .filter((habit) => habit.period === 'day' && isHabitScheduled(habit, today))
    .map((habit) => ({ ...periodStatus(state, habit, today), previousMisses: previousMisses(state, habit, today) }))
    .filter((status) => !status.complete);
  const flexible = active
    .filter((habit) => habit.period !== 'day')
    .map((habit) => periodStatus(state, habit, today))
    .filter((status) => !status.complete);
  const overdue = due.filter((status) => status.previousMisses > 0);
  return { due, flexible, overdue };
}

function rankTasks(state: SlateState | null, todayKey: DateKey): {
  priority: PriorityTask | null;
  due: SlateTask[];
  open: number;
} {
  if (!state) return { priority: null, due: [], open: 0 };
  const sections = new Map(state.sections.filter((section) => !section.deleted).map((section) => [section.id, section]));
  const tasks = state.tasks.filter((task) => !task.deleted && !task.done);
  const ranked = [...tasks].sort((left, right) => {
    const score = (task: SlateTask) => {
      if (task.due && task.due < todayKey) return 0;
      if (task.due === todayKey) return 1;
      if (task.due) return 2;
      return 3;
    };
    const scoreDifference = score(left) - score(right);
    if (scoreDifference) return scoreDifference;
    const dueDifference = (left.due ?? '9999-99-99').localeCompare(right.due ?? '9999-99-99');
    if (dueDifference) return dueDifference;
    const sectionDifference = (sections.get(left.sectionId)?.order ?? 0) - (sections.get(right.sectionId)?.order ?? 0);
    return sectionDifference || left.order - right.order || left.createdAt.localeCompare(right.createdAt);
  });
  const task = ranked[0];
  let priority: PriorityTask | null = null;
  if (task) {
    const reason = task.due && task.due < todayKey
      ? `Overdue since ${formatShortDate(task.due)}`
      : task.due === todayKey
        ? 'Due today'
        : task.due
          ? `Due ${formatShortDate(task.due)}`
          : 'First in Slate';
    priority = { task, section: sections.get(task.sectionId)?.title ?? 'Tasks', reason };
  }
  const due = ranked.filter((task) => task.due && task.due <= todayKey);
  return { priority, due, open: ranked.length };
}

function deriveNutrition(state: FareState | null, dateKey: DateKey): NutritionSummary | null {
  if (!state) return null;
  const entries = state.entries.filter((entry) => !entry.deleted && entry.dateKey === dateKey);
  const consumedCalories = entries.reduce((sum, entry) => sum + Math.max(0, Number(entry.snapshot?.nutrition?.calories) || 0), 0);
  const consumedProtein = entries.reduce((sum, entry) => sum + Math.max(0, Number(entry.snapshot?.nutrition?.proteinG) || 0), 0);
  return {
    consumedCalories,
    consumedProtein,
    targetCalories: Math.max(0, Number(state.targets.calories) || 0),
    targetProtein: Math.max(0, Number(state.targets.proteinG) || 0),
    remainingCalories: Math.max(0, (Number(state.targets.calories) || 0) - consumedCalories),
    remainingProtein: Math.max(0, (Number(state.targets.proteinG) || 0) - consumedProtein),
    entries: entries.length,
  };
}

function weekday(date: Date): string {
  return ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'][date.getDay()];
}

function deriveWorkout(sources: TodaySources, today: Date, dateKey: DateKey): WorkoutSummary | null {
  const gym = sources.gym.state;
  if (!gym) return null;
  const log = gym.logs[dateKey] ?? null;
  const exercises = log?.exerciseSnapshot?.length
    ? log.exerciseSnapshot
    : gym.program?.[weekday(today)] ?? [];
  const grouped = new Map<string, Exercise[]>();
  exercises.forEach((exercise) => {
    const label = exercise.workoutLabel || 'Today’s workout';
    grouped.set(label, [...(grouped.get(label) ?? []), exercise]);
  });
  return {
    exercises,
    blocks: [...grouped.entries()].map(([label, blockExercises]) => ({ label, exercises: blockExercises })),
    log,
    completed: log?.completed.filter((id) => exercises.some((exercise) => exercise.id === id)).length ?? 0,
    skipped: log?.skipped.filter((id) => exercises.some((exercise) => exercise.id === id)).length ?? 0,
    courtSports: exercises.filter((exercise) => /basketball|badminton/i.test(exercise.name)),
  };
}

function adherence(value: number, target: number): number {
  if (!target) return 0;
  const ratio = value / target;
  return Math.max(0, Math.min(1, ratio <= 1 ? ratio : 2 - ratio));
}

function dailyNutritionScore(state: FareState, key: DateKey): number | null {
  const entries = state.entries.filter((entry) => !entry.deleted && entry.dateKey === key);
  if (!entries.length) return null;
  const calories = entries.reduce((sum, entry) => sum + Math.max(0, Number(entry.snapshot?.nutrition?.calories) || 0), 0);
  const protein = entries.reduce((sum, entry) => sum + Math.max(0, Number(entry.snapshot?.nutrition?.proteinG) || 0), 0);
  return (adherence(calories, state.targets.calories) + adherence(protein, state.targets.proteinG)) / 2;
}

function dailyHabitScore(state: DaymarkState, date: Date): number | null {
  const eligible = state.habits.filter((habit) => {
    if (!isHabitScheduled(habit, date)) return false;
    const entry = entryFor(state, habit, date);
    return habit.period === 'day' || Boolean(entry && !entry.skipped && entry.hasValue !== false);
  });
  if (!eligible.length) return null;
  const values = eligible.map((habit) => {
    const entry = entryFor(state, habit, date);
    if (entry?.skipped) return null;
    if (!entry || entry.hasValue === false) return 0;
    if (habit.period === 'day') return periodStatus(state, habit, date).ratio;
    if (habit.direction === 'atMost') return entry.value <= habit.target ? 1 : Math.min(1, habit.target / Math.max(entry.value, 1));
    if (habit.metric === 'check') return entry.value > 0 ? 1 : 0;
    return Math.min(1, entry.value / Math.max(habit.increment, 0.0001));
  }).filter((value): value is number => value !== null);
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function dailyTrainingScore(log: WorkoutLog | undefined): number | null {
  if (!log) return null;
  if (log.daySkipped) return 0;
  const total = new Set([...log.completed, ...log.skipped, ...(log.exerciseSnapshot?.map((exercise) => exercise.id) ?? [])]).size;
  return total ? Math.min(1, log.completed.length / total) : 0;
}

function average(values: Array<number | null>): number | null {
  const present = values.filter((value): value is number => value !== null && Number.isFinite(value));
  return present.length ? present.reduce((sum, value) => sum + value, 0) / present.length : null;
}

function deriveTrends(sources: TodaySources, today: Date): TrendWeek[] {
  return [-3, -2, -1, 0].map((weekOffset) => {
    const end = addDays(today, weekOffset * 7);
    const start = addDays(end, -6);
    const dates = daysBetween(start, end);
    return {
      label: weekOffset === 0 ? 'Latest' : formatShortDate(toDateKey(start)),
      nutrition: sources.fare.state ? average(dates.map((date) => dailyNutritionScore(sources.fare.state!, toDateKey(date)))) : null,
      training: sources.gym.state ? average(dates.map((date) => dailyTrainingScore(sources.gym.state!.logs[toDateKey(date)]))) : null,
      habits: sources.daymark.state ? average(dates.map((date) => dailyHabitScore(sources.daymark.state!, date))) : null,
    };
  });
}

function direction(value: number): 'up' | 'down' | 'steady' {
  if (value > 0.04) return 'up';
  if (value < -0.04) return 'down';
  return 'steady';
}

function trendNarrative(trends: TrendWeek[]): string {
  const previous = trends[2];
  const current = trends[3];
  const metrics = [
    ['Nutrition', previous.nutrition, current.nutrition],
    ['Training', previous.training, current.training],
    ['Habits', previous.habits, current.habits],
  ] as const;
  const available = metrics.filter(([, before, after]) => before !== null && after !== null);
  if (available.length < 2) return 'A second week of logs will reveal whether nutrition, training, and habits are moving together.';
  const movements = available.map(([name, before, after]) => ({ name, movement: direction((after ?? 0) - (before ?? 0)) }));
  const groups = new Map<string, string[]>();
  movements.forEach(({ name, movement }) => groups.set(movement, [...(groups.get(movement) ?? []), name.toLowerCase()]));
  const strongest = [...groups.entries()].sort((left, right) => right[1].length - left[1].length)[0];
  if (strongest[1].length >= 2) {
    const verb = strongest[0] === 'up' ? 'improved' : strongest[0] === 'down' ? 'softened' : 'held steady';
    const names = strongest[1].length === 2 ? strongest[1].join(' and ') : `${strongest[1].slice(0, -1).join(', ')}, and ${strongest[1].at(-1)}`;
    const outliers = movements.filter(({ movement }) => movement !== strongest[0]).map(({ name }) => name.toLowerCase());
    return `${capitalize(names)} ${verb} together${outliers.length ? `; ${outliers.join(' and ')} moved differently` : ''}.`;
  }
  return 'The three signals moved independently this week; no shared direction is clear yet.';
}

function derivePressure(options: {
  sources: TodaySources;
  dueTasks: SlateTask[];
  openTasks: number;
  dueHabits: HabitStatus[];
  overdueHabits: HabitStatus[];
  nutrition: NutritionSummary | null;
  workout: WorkoutSummary | null;
  now: Date;
}): DayPressure {
  const minute = options.now.getHours() * 60 + options.now.getMinutes();
  // Slate retired its schedule blocks, so no source on this device knows how
  // much of the day is already committed. This is deliberately a load-only
  // estimate: open tasks contribute workload, never invented availability.
  // Tasks already due are scored above and excluded from the backlog count.
  const backlogTasks = Math.max(0, options.openTasks - options.dueTasks.length);
  const openWorkout = options.workout
    ? Math.max(0, options.workout.exercises.length - options.workout.completed - options.workout.skipped)
    : 0;
  const lateProteinGap = options.nutrition && options.nutrition.targetProtein > 0 && minute >= 15 * 60
    ? options.nutrition.remainingProtein / options.nutrition.targetProtein
    : 0;
  const score = Math.min(100, Math.round(
    Math.min(28, options.dueTasks.length * 9)
    + Math.min(22, options.overdueHabits.reduce((sum, habit) => sum + habit.previousMisses * 3, 0))
    + Math.min(10, options.dueHabits.length * 2.5)
    + Math.min(20, backlogTasks * 1.5)
    + Math.min(12, openWorkout * 1.5)
    + Math.min(8, lateProteinGap * 8),
  ));
  const label = score >= 70 ? 'Tight' : score >= 50 ? 'Loaded' : score >= 25 ? 'Manageable' : 'Light';
  const connectedCount = Object.values(options.sources).filter((source) => source.connected).length;
  // Even with all four sources connected, missing calendar commitments cap the
  // estimate at medium confidence. A future schedule source can restore high.
  const confidence = connectedCount >= 2 ? 'Medium' : 'Low';
  const factors = [
    options.dueTasks.length ? `${options.dueTasks.length} due task${options.dueTasks.length === 1 ? '' : 's'}` : '',
    options.overdueHabits.length ? `${options.overdueHabits.length} habit${options.overdueHabits.length === 1 ? '' : 's'} carrying misses` : '',
    backlogTasks ? `${backlogTasks} other open task${backlogTasks === 1 ? '' : 's'} in Slate` : '',
    openWorkout ? `${openWorkout} workout movement${openWorkout === 1 ? '' : 's'} open` : '',
  ].filter(Boolean).slice(0, 3);
  return { score, label, confidence, basis: 'load-only', factors };
}

function trendDelta(trends: TrendWeek[], key: 'nutrition' | 'training' | 'habits'): number | null {
  const before = trends[2][key];
  const after = trends[3][key];
  return before === null || after === null ? null : after - before;
}

function deriveAdvice(options: {
  now: Date;
  priority: PriorityTask | null;
  dueTasks: SlateTask[];
  overdueHabits: HabitStatus[];
  flexibleHabits: HabitStatus[];
  nutrition: NutritionSummary | null;
  workout: WorkoutSummary | null;
  trends: TrendWeek[];
  weekStartsOn: 0 | 1;
}): AdviceInsight[] {
  const advice: AdviceInsight[] = [];
  const minute = options.now.getHours() * 60 + options.now.getMinutes();
  const add = (insight: AdviceInsight) => advice.push(insight);

  if (options.priority && options.priority.reason.startsWith('Overdue')) {
    const alsoDue = Math.max(0, options.dueTasks.length - 1);
    add({
      id: 'clear-overdue-task',
      title: `Move “${options.priority.task.title}” first`,
      detail: alsoDue
        ? `Close it before the ${alsoDue} other task${alsoDue === 1 ? '' : 's'} due today, and before adding anything new.`
        : 'Give it the next unclaimed stretch of the day before adding anything new.',
      evidence: `${options.priority.reason}; Slate ranks it above the remaining open work.`,
      source: 'Slate', href: '/slate/', priority: 96, tone: 'act',
    });
  } else if (options.priority) {
    add({
      id: 'start-priority-task',
      title: `Start “${options.priority.task.title}”`,
      detail: options.dueTasks.length
        ? `${options.dueTasks.length} task${options.dueTasks.length === 1 ? ' is' : 's are'} due today and this one ranks first.`
        : 'Nothing is due yet, so the top of the open list is the cheapest thing to finish next.',
      evidence: `Slate ranks it first in ${options.priority.section} (${options.priority.reason.toLowerCase()}).`,
      source: 'Slate', href: '/slate/', priority: 82, tone: 'act',
    });
  }

  if (options.overdueHabits.length) {
    const totalMisses = options.overdueHabits.reduce((sum, habit) => sum + habit.previousMisses, 0);
    const first = options.overdueHabits[0];
    add({
      id: 'recover-habit',
      title: `Restart with ${first.habit.name}`,
      detail: 'Complete one clean repetition today; do not try to repay every missed day.',
      evidence: `${options.overdueHabits.length} daily habit${options.overdueHabits.length === 1 ? '' : 's'} account for ${totalMisses} unresolved scheduled occurrence${totalMisses === 1 ? '' : 's'}.`,
      source: 'Daymark', href: '/daymark/', priority: 78, tone: 'protect',
    });
  }

  if (options.flexibleHabits.length) {
    const endingSoon = options.flexibleHabits.filter((status) => {
      const end = endOfPeriod(status.habit.period, options.now, options.weekStartsOn);
      return Math.round((end.getTime() - fromDateKey(toDateKey(options.now)).getTime()) / 86_400_000) <= 1;
    });
    if (endingSoon.length) {
      add({
        id: 'close-flex-window',
        title: `Close the flexible window for ${endingSoon[0].habit.name}`,
        detail: `${endingSoon[0].label}; its ${endingSoon[0].habit.period} window is nearly over.`,
        evidence: `${endingSoon.length} flexible goal${endingSoon.length === 1 ? '' : 's'} reach the end of their period within one day.`,
        source: 'Daymark', href: '/daymark/', priority: 74, tone: 'prepare',
      });
    }
  }

  if (options.nutrition && options.nutrition.targetProtein > 0) {
    const proteinGap = options.nutrition.remainingProtein / options.nutrition.targetProtein;
    const calorieOver = options.nutrition.consumedCalories > options.nutrition.targetCalories;
    if (proteinGap >= 0.35 && minute >= 13 * 60) {
      add({
        id: 'protein-gap',
        title: calorieOver ? 'Favor protein without adding much energy' : 'Make the next meal protein-forward',
        detail: `${formatNumber(options.nutrition.remainingProtein)} g protein remains${options.nutrition.remainingCalories ? ` alongside ${formatNumber(options.nutrition.remainingCalories)} kcal` : ''}.`,
        evidence: `${Math.round(proteinGap * 100)}% of the Fare protein target is still open.`,
        source: 'Fare', href: '/fare/', priority: options.workout?.exercises.length ? 76 : 68, tone: 'prepare',
      });
    } else if (options.nutrition.entries > 0 && options.nutrition.remainingProtein <= 0) {
      add({
        id: 'protein-covered',
        title: 'Protein target is covered',
        detail: 'No nutrition correction is needed from this dashboard right now.',
        evidence: `Fare shows ${formatNumber(options.nutrition.consumedProtein)} g against a ${formatNumber(options.nutrition.targetProtein)} g target.`,
        source: 'Fare', href: '/fare/', priority: 35, tone: 'positive',
      });
    }
  }

  if (options.workout?.courtSports.length) {
    const sport = options.workout.courtSports.map((exercise) => exercise.name.replace(/\s+minutes/i, ' min')).join(' + ');
    add({
      id: 'court-plan',
      title: `Keep room for ${sport}`,
      detail: options.workout.completed ? 'Some training is already done; keep the remaining court time visible when the day gets busy.' : 'It is part of today’s Gym program, so protect the time it needs.',
      evidence: `${options.workout.courtSports.length} court session${options.workout.courtSports.length === 1 ? '' : 's'} detected in today’s Gym plan.`,
      source: 'Gym', href: '/gym/', priority: 72, tone: 'prepare',
    });
  } else if (options.workout && options.workout.exercises.length && options.workout.completed >= options.workout.exercises.length - options.workout.skipped) {
    add({
      id: 'workout-complete',
      title: 'Training is handled',
      detail: 'Leave the workout closed and spend the remaining attention elsewhere.',
      evidence: `Gym shows ${options.workout.completed} of ${options.workout.exercises.length} movements complete.`,
      source: 'Gym', href: '/gym/', priority: 38, tone: 'positive',
    });
  }

  const nutritionDelta = trendDelta(options.trends, 'nutrition');
  const trainingDelta = trendDelta(options.trends, 'training');
  const habitDelta = trendDelta(options.trends, 'habits');
  const declining = [nutritionDelta, trainingDelta, habitDelta].filter((delta) => delta !== null && delta < -0.08).length;
  if (declining >= 2) {
    add({
      id: 'trend-reset',
      title: 'Reduce the plan to the minimum viable day',
      detail: 'Choose one task, one habit, and the planned training anchor instead of spreading effort thin.',
      evidence: `${declining} of the three 7-day consistency signals declined by more than 8 points together.`,
      source: 'Today', href: '#trend-card', priority: 88, tone: 'protect',
    });
  }

  if (!advice.length) {
    add({
      id: 'steady-day',
      title: 'Keep the day simple',
      detail: 'Nothing in the connected sources requires a correction. Continue with the highest-priority open item.',
      evidence: 'No high-pressure rule fired from the available task, habit, nutrition, or training data.',
      source: 'Today', href: '#source-strip', priority: 20, tone: 'positive',
    });
  }

  return advice.sort((left, right) => right.priority - left.priority).slice(0, 3);
}

function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

export function formatShortDate(key: DateKey): string {
  return new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric' }).format(fromDateKey(key));
}

export function deriveDashboard(sources: TodaySources, now = new Date()): DashboardModel {
  const todayKey = toDateKey(now);
  const tasks = rankTasks(sources.slate.state, todayKey);
  const habits = deriveHabits(sources.daymark.state, now);
  const trends = deriveTrends(sources, now);
  const nutrition = deriveNutrition(sources.fare.state, todayKey);
  const workout = deriveWorkout(sources, now, todayKey);
  const pressure = derivePressure({
    sources,
    dueTasks: tasks.due,
    openTasks: tasks.open,
    dueHabits: habits.due,
    overdueHabits: habits.overdue,
    nutrition,
    workout,
    now,
  });
  const advice = deriveAdvice({
    now,
    priority: tasks.priority,
    dueTasks: tasks.due,
    overdueHabits: habits.overdue,
    flexibleHabits: habits.flexible,
    nutrition,
    workout,
    trends,
    weekStartsOn: sources.daymark.state?.profile.weekStartsOn ?? 1,
  });
  return {
    now,
    todayKey,
    priority: tasks.priority,
    dueTasks: tasks.due,
    openTasks: tasks.open,
    dueHabits: habits.due,
    flexibleHabits: habits.flexible,
    overdueHabits: habits.overdue,
    nutrition,
    workout,
    trends,
    trendNarrative: trendNarrative(trends),
    advice,
    pressure,
    connectedCount: Object.values(sources).filter((source) => source.connected).length,
    sources,
  };
}
