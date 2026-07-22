import assert from 'node:assert/strict';

import {
    DEFAULT_SETTINGS,
    getOpenAIProxyBase,
    injectFunctionResponsePrefill,
} from '../lib.js';

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

assert.match(functionPrefill.text, /^BETA\.?$/);

const trailingRequest = {
    type: 'normal',
    model,
    chat_completion_source: 'makersuite',
    reverse_proxy: baseUrl,
    frequency_penalty: 0.75,
    presence_penalty: 0.5,
    messages: [{ role: 'assistant', content: 'Reply with exactly OK.' }],
    custom_prompt_post_processing: '',
};
const trailingResult = injectFunctionResponsePrefill(
    trailingRequest,
    {
        ...DEFAULT_SETTINGS,
        captureTrailingAssistantPrefill: true,
        useStartReplyWith: true,
        useOpenAIProxyTransport: true,
    },
    'Assistant reply: ',
    { idFactory: () => 'sfr_trailing_smoke_test' },
);
assert.equal(trailingResult.injected, true);
assert.equal(trailingResult.prefillSource, 'trailing-assistant+start-reply-with');
assert.equal(trailingResult.trailingPrefillRemoved, true);
assert.equal(trailingRequest.messages.at(-1).content, 'Reply with exactly OK.\n\nAssistant reply: ');
assert.equal(trailingRequest.chat_completion_source, 'openai');
assert.equal(Object.hasOwn(trailingRequest, 'frequency_penalty'), false);
assert.equal(Object.hasOwn(trailingRequest, 'presence_penalty'), false);

const trailingPrefill = await complete(trailingRequest.messages);
assert.match(trailingPrefill.text, /^OK\.?$/);

console.log(JSON.stringify({
    model,
    basic: basic.text,
    directAssistantPrefill: directPrefill.text,
    functionResponsePrefill: functionPrefill.text,
    capturedTrailingAssistantPrefill: trailingPrefill.text,
}, null, 2));
