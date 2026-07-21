import assert from 'node:assert/strict';

import { getOpenAIProxyBase } from '../lib.js';

const baseUrl = String(process.env.SFR_PROXY_URL ?? '').replace(/\/$/, '');
const apiKey = String(process.env.SFR_PROXY_KEY ?? '');
const model = String(process.env.SFR_MODEL ?? 'gemini-3.6-flash');

if (!baseUrl || !apiKey) {
    console.error('Set SFR_PROXY_URL and SFR_PROXY_KEY before running this smoke test.');
    process.exit(2);
}

async function complete(messages) {
    const response = await fetch(`${getOpenAIProxyBase(baseUrl)}/chat/completions`, {
        method: 'POST',
        headers: {
            Authorization: `Bearer ${apiKey}`,
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model,
            messages,
            max_tokens: 1024,
            temperature: 0,
            stream: false,
            reasoning_effort: 'minimal',
        }),
    });

    const text = await response.text();
    assert.equal(response.ok, true, `HTTP ${response.status}: ${text}`);

    if (!text) {
        return { text: '', body: null };
    }

    const body = JSON.parse(text);
    return {
        text: String(body?.choices?.[0]?.message?.content ?? ''),
        body,
    };
}

const basic = await complete([{ role: 'user', content: 'Reply with exactly: OK' }]);
assert.equal(basic.text, 'OK');

const directPrefill = await complete([
    { role: 'user', content: 'Complete the exact string ALPHA-BETA.' },
    { role: 'assistant', content: 'ALPHA-' },
]);

const functionPrefill = await complete([
    { role: 'user', content: 'Complete the exact string ALPHA-BETA.' },
    {
        role: 'assistant',
        content: '',
        tool_calls: [{
            id: 'sfr_smoke_test',
            type: 'function',
            function: {
                name: 'silly_prefill',
                arguments: '{"mode":"assistant_prefill"}',
            },
        }],
    },
    {
        role: 'tool',
        name: 'silly_prefill',
        tool_call_id: 'sfr_smoke_test',
        content: 'ALPHA-',
    },
]);

assert.equal(functionPrefill.text, 'BETA');

console.log(JSON.stringify({
    model,
    basic: basic.text,
    directAssistantPrefill: directPrefill.text,
    functionResponsePrefill: functionPrefill.text,
}, null, 2));
