export type DateKey = string;
export type GoalPeriod = 'day' | 'week' | 'month';
export type GoalDirection = 'atLeast' | 'atMost';
export type TimeSlot = 'morning' | 'anytime' | 'evening';

export interface HabitSchedule {
  type: 'everyday' | 'selectedDays' | 'interval';
  days?: number[];
  every?: number;
  unit?: 'day' | 'week';
}

export interface Habit {
  id: string;
  name: string;
  metric: 'check' | 'count' | 'duration' | 'quantity' | 'distance';
  target: number;
  unit: string;
  period: GoalPeriod;
  direction: GoalDirection;
  schedule: HabitSchedule;
  timeSlot: TimeSlot;
  increment: number;
  startDate: DateKey;
  archivedAt?: string;
  pauses?: Array<{ start: DateKey; end?: DateKey }>;
}

export interface HabitEntry {
  value: number;
  hasValue?: boolean;
  skipped?: boolean;
}

export interface DaymarkState {
  profile: { weekStartsOn: 0 | 1 };
  habits: Habit[];
  entries: Record<DateKey, Record<string, HabitEntry>>;
}

export interface SlateTask {
  id: string;
  sectionId: string;
  title: string;
  notes: string;
  done: boolean;
  due?: DateKey;
  order: number;
  createdAt: string;
  deleted?: boolean;
}

export interface SlateSection {
  id: string;
  title: string;
  order: number;
  deleted?: boolean;
}

export interface SlateBlock {
  id: string;
  dateKey: DateKey;
  startMin: number;
  durationMin: number;
  title: string;
  color: string;
  deleted?: boolean;
}

export interface SlateState {
  tasks: SlateTask[];
  sections: SlateSection[];
  blocks: SlateBlock[];
}

export interface NutritionValue {
  calories: number;
  proteinG: number;
}

export interface FareEntry {
  id: string;
  dateKey: DateKey;
  snapshot: { nutrition: NutritionValue };
  deleted?: boolean;
}

export interface FareState {
  profile: { onboardingComplete: boolean };
  targets: NutritionValue;
  entries: FareEntry[];
}

export interface Exercise {
  id: string;
  name: string;
  kind: 'strength' | 'cardio' | 'mobility';
  target?: { sets?: number; minutes?: number };
  workoutBlock?: 1 | 2;
  workoutLabel?: string;
}

export interface WorkoutLog {
  date: DateKey;
  completed: string[];
  skipped: string[];
  daySkipped: boolean;
  exerciseSnapshot?: Exercise[];
}

export type ProgramByDay = Record<string, Exercise[]>;
export type LogsByDate = Record<DateKey, WorkoutLog>;

export interface AppData<T> {
  state: T | null;
  connected: boolean;
  source: 'indexeddb' | 'localstorage' | 'none';
}

export interface TodaySources {
  daymark: AppData<DaymarkState>;
  slate: AppData<SlateState>;
  fare: AppData<FareState>;
  gym: AppData<{ program: ProgramByDay | null; logs: LogsByDate }>;
}

export interface HabitStatus {
  habit: Habit;
  value: number;
  target: number;
  ratio: number;
  complete: boolean;
  label: string;
  previousMisses: number;
}

export interface TrendWeek {
  label: string;
  nutrition: number | null;
  training: number | null;
  habits: number | null;
}
