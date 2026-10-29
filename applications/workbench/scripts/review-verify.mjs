import assert from 'node:assert/strict';
import { database } from '../tests/d1-adapter.mjs';
import { handleApi } from '../lib/api.mjs';

// Exercise the same API handler against isolated local storage. This does not
// simulate the hosted sign-in gateway or measure network performance.
const db = database();
const origin = 'https://fraudscope.test';
async function request(path, body, owner = 'reviewer') {
  return handleApi(new Request(`${origin}/api${path}`, {
    method: body === undefined ? 'GET' : 'POST',
    headers: { 'oai-authenticated-user-id': owner,
      ...(body === undefined ? {} : { 'Content-Type': 'application/json', Origin: origin }) },
    body: body === undefined ? undefined : JSON.stringify(body),
  }), db);
}
async function json(path, body, owner) {
  const response = await request(path, body, owner);
  assert.equal(response.status, 200, `${path}: ${await response.clone().text()}`);
  return response.json();
}
try {
  await json('/workspace', {});
  for (let i=0;i<6;i++) assert.equal((await json('/replay',{count:4})).scored,4);
  assert.equal((await json('/overview')).totals.events,24);
  const item=await json('/transactions/demo_00024');
  assert.ok(item.result.graph.every(r=>r.neighbors.every(n=>n.age_hours>0)));
  assert.equal((await json('/transactions/demo_00024/models')).models.length,4);
  assert.equal((await json('/score',item.event)).idempotent_replay,true);
  assert.equal((await request('/score',{...item.event,amount:item.event.amount+1})).status,409);
  await json('/workspace',{},'other-reviewer');
  assert.equal((await json('/transactions',undefined,'other-reviewer')).items.length,0);
  assert.equal((await request('/transactions/demo_00024',undefined,'other-reviewer')).status,404);
  const review={verdict:'needs_info',notes:'Synthetic demonstration only. Analyst assessment is unverified.',expected_revision:0};
  assert.equal((await json('/transactions/demo_00024/review',review)).revision,1);
  assert.equal((await request('/transactions/demo_00024/review',review)).status,409);
  const csv=await (await request('/export')).text();
  assert.equal(csv.trim().split(/\r?\n/).length,25);
  assert.ok(csv.includes(review.notes));
  const audit=await json('/audit/verify',{});
  assert.equal(audit.valid,true);assert.equal(audit.checked,25);
  console.log(JSON.stringify({events:24,neural_comparisons:4,audit_records:25,
    exact_retry:true,conflicting_retry_rejected:true,workspace_isolation:true,
    stale_review_rejected:true,csv_rows:24,storage:'isolated SQLite',hosted_browser_test:false},null,2));
  console.log('REVIEWER TOUR PASSED');
} finally {db.close();}
