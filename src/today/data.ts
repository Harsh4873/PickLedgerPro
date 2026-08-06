import type {
  AppData,
  DaymarkState,
  FareState,
  LogsByDate,
  ProgramByDay,
  SlateState,
  TodaySources,
} from './types';

interface Candidate<T> {
  state: T;
  savedAt: number;
  source: 'indexeddb' | 'localstorage';
}

function object(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null ? value as Record<string, unknown> : null;
}

function savedAt(value: unknown): number {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string') {
    const parsed = Date.parse(value);
    return Number.isFinite(parsed) ? parsed : 0;
  }
  return 0;
}

function envelopeState<T>(value: unknown): { state: T; savedAt: number } | null {
  const raw = object(value);
  if (!raw) return null;
  const state = object(raw.state) ?? raw;
  return { state: state as T, savedAt: savedAt(raw.savedAt) };
}

function localCandidate<T>(key: string): Candidate<T> | null {
  try {
    const text = window.localStorage.getItem(key);
    if (!text) return null;
    const parsed = envelopeState<T>(JSON.parse(text));
    return parsed ? { ...parsed, source: 'localstorage' } : null;
  } catch {
    return null;
  }
}

function readIndexedValue(databaseName: string, storeName: string, key: string): Promise<unknown> {
  return new Promise((resolve) => {
    if (!('indexedDB' in window)) {
      resolve(null);
      return;
    }
    let request: IDBOpenDBRequest;
    try {
      request = window.indexedDB.open(databaseName);
    } catch {
      resolve(null);
      return;
    }
    // A read must never create an empty source-app database. Aborting the
    // creation upgrade keeps Today a strictly read-only integration surface.
    request.onupgradeneeded = () => request.transaction?.abort();
    request.onerror = () => resolve(null);
    request.onblocked = () => resolve(null);
    request.onsuccess = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(storeName)) {
        database.close();
        resolve(null);
        return;
      }
      try {
        const transaction = database.transaction(storeName, 'readonly');
        const read = transaction.objectStore(storeName).get(key);
        read.onsuccess = () => resolve(read.result ?? null);
        read.onerror = () => resolve(null);
        transaction.oncomplete = () => database.close();
        transaction.onabort = () => database.close();
      } catch {
        database.close();
        resolve(null);
      }
    };
  });
}

function validDaymark(value: unknown): value is DaymarkState {
  const raw = object(value);
  return Boolean(raw && object(raw.profile) && Array.isArray(raw.habits) && object(raw.entries));
}

function validSlate(value: unknown): value is SlateState {
  const raw = object(value);
  return Boolean(raw && Array.isArray(raw.tasks) && Array.isArray(raw.sections) && Array.isArray(raw.blocks));
}

function validFare(value: unknown): value is FareState {
  const raw = object(value);
  return Boolean(raw && object(raw.profile) && object(raw.targets) && Array.isArray(raw.entries));
}

async function mirroredState<T>(options: {
  localKey: string;
  database: string;
  store: string;
  key: string;
  validate: (value: unknown) => value is T;
}): Promise<AppData<T>> {
  const candidates: Candidate<T>[] = [];
  const local = localCandidate<T>(options.localKey);
  if (local && options.validate(local.state)) candidates.push(local);

  const indexed = envelopeState<T>(await readIndexedValue(options.database, options.store, options.key));
  if (indexed && options.validate(indexed.state)) {
    candidates.push({ ...indexed, savedAt: indexed.savedAt + 0.5, source: 'indexeddb' });
  }
  candidates.sort((left, right) => right.savedAt - left.savedAt);
  const chosen = candidates[0];
  return chosen
    ? { state: chosen.state, connected: true, source: chosen.source }
    : { state: null, connected: false, source: 'none' };
}

function parseGymStorage<T>(key: string, nestedKey: string): T | null {
  try {
    const text = window.localStorage.getItem(key);
    if (!text) return null;
    const parsed = object(JSON.parse(text));
    if (!parsed) return null;
    return (parsed[nestedKey] ?? parsed) as T;
  } catch {
    return null;
  }
}

function readGym(): AppData<{ program: ProgramByDay | null; logs: LogsByDate }> {
  const program = parseGymStorage<ProgramByDay>('harsh-gym-program-v1', 'program');
  const logs = parseGymStorage<LogsByDate>('harsh-gym-logs-v1', 'logs') ?? {};
  const connected = Boolean(program || Object.keys(logs).length);
  return {
    state: connected ? { program, logs } : null,
    connected,
    source: connected ? 'localstorage' : 'none',
  };
}

export async function readTodaySources(): Promise<TodaySources> {
  const [daymark, slate, fare] = await Promise.all([
    mirroredState({
      localKey: 'daymark-tracker-state-v2',
      database: 'daymark-tracker',
      store: 'tracker-state',
      key: 'current',
      validate: validDaymark,
    }),
    mirroredState({
      localKey: 'slate-todo-state-v1',
      database: 'slate-todo',
      store: 'slate-state',
      key: 'current',
      validate: validSlate,
    }),
    mirroredState({
      localKey: 'fare-state-v1',
      database: 'fare-local',
      store: 'state',
      key: 'current',
      validate: validFare,
    }),
  ]);

  return { daymark, slate, fare, gym: readGym() };
}
