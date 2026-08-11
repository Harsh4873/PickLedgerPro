import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { envelopeState, validSlate } from './data.ts';
import { deriveDashboard } from './derive.ts';
import type { SlateState, TodaySources } from './types.ts';

/**
 * The payload below is not written by hand: it is generated in the Slate repo
 * by `buildStorageEnvelope` (src/store.ts) and snapshotted by its own tests as
 * `tests/fixtures/today-slate-payload.json`. Copy that file over this one when
 * Slate's storage shape changes — both repos' tests fail until it matches.
 *
 * Today shipped for weeks reading nothing from Slate because the validator
 * asked for a top-level `tasks` array plus a `blocks` array from Slate's
 * retired schedule feature. Neither has ever existed in this payload.
 */
const payload: Record<string, unknown> = JSON.parse(
  readFileSync(new URL('../../tests/fixtures/slate-persisted-payload.json', import.meta.url), 'utf8'),
);

test('the fixture really is what Slate persists: an envelope with no top-level tasks and no blocks', () => {
  assert.equal(payload.storageFormat, 'slate-v1');
  assert.equal('tasks' in payload, false);
  assert.equal('sections' in payload, false);
  const state = payload.state as Record<string, unknown>;
  assert.equal('blocks' in state, false);
  assert.ok(Array.isArray(state.tasks) && state.tasks.length > 0);
});

test('Slate reads as connected from its own persisted payload', () => {
  const unwrapped = envelopeState<SlateState>(payload);
  assert.ok(unwrapped, 'the envelope should unwrap to Slate’s nested state');
  assert.equal(validSlate(unwrapped.state), true);
  assert.ok(unwrapped.savedAt > 0, 'savedAt should come from the envelope');
});

test('the envelope itself is rejected, so only the nested state can validate', () => {
  assert.equal(validSlate(payload), false);
});

test('a bare state without the envelope still validates', () => {
  const bare = envelopeState<SlateState>(payload.state);
  assert.ok(bare);
  assert.equal(validSlate(bare.state), true);
  assert.equal(bare.savedAt, 0);
});

test('unreadable or foreign payloads stay disconnected', () => {
  assert.equal(validSlate(null), false);
  assert.equal(validSlate({ version: 2, tasks: [], sections: [] }), false);
  assert.equal(validSlate({ version: 1, tasks: [] }), false);
  assert.equal(validSlate({ version: 1, sections: [] }), false);
});

test('the dashboard derives real work from the payload', () => {
  const unwrapped = envelopeState<SlateState>(payload);
  assert.ok(unwrapped && validSlate(unwrapped.state));
  const sources: TodaySources = {
    daymark: { state: null, connected: false, source: 'none' },
    slate: { state: unwrapped.state, connected: true, source: 'localstorage' },
    fare: { state: null, connected: false, source: 'none' },
    gym: { state: null, connected: false, source: 'none' },
  };

  // The fixture holds one task due 2026-07-12, one undated, one completed.
  const model = deriveDashboard(sources, new Date(2026, 6, 12, 9, 0));
  assert.equal(model.connectedCount, 1);
  assert.equal(model.openTasks, 2);
  assert.equal(model.dueTasks.length, 1);
  assert.equal(model.priority?.task.id, 'task-reading');
  assert.equal(model.priority?.reason, 'Due today');
  assert.equal(model.priority?.section, 'Inbox');
  assert.ok(model.pressure.score > 0, 'due work should register as pressure');
});
