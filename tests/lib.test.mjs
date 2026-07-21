import assert from 'node:assert/strict';
import test from 'node:test';

import {
    DEFAULT_SETTINGS,
    TOOL_NAME,
    getOpenAIProxyBase,
    injectFunctionResponsePrefill,
    isGeminiModel,
    mergeSettings,
    preserveToolMessages,
    removeTrailingAssistantPrefill,
    routeNativeGeminiProxyThroughOpenAI,
} from '../lib.js';

function request(overrides = {}) {
    return {
        type: 'normal',
        model: 'gemini-3.6-flash',
        messages: [{ role: 'user', content: 'Hello' }],
        custom_prompt_post_processing: '',
        ...overrides,
    };
}

test('mergeSettings fills new defaults and preserves saved values', () => {
    const merged = mergeSettings({ enabled: false, prefill: 'x' });
    assert.equal(merged.enabled, false);
    assert.equal(merged.prefill, 'x');
    assert.equal(merged.useStartReplyWith, DEFAULT_SETTINGS.useStartReplyWith);
    assert.equal(merged.geminiOnly, DEFAULT_SETTINGS.geminiOnly);
    assert.equal(merged.useOpenAIProxyTransport, false);
});

test('isGeminiModel accepts direct and routed Gemini identifiers', () => {
    assert.equal(isGeminiModel('gemini-3.6-flash'), true);
    assert.equal(isGeminiModel('google/gemini-3.6-flash'), true);
    assert.equal(isGeminiModel('gpt-5.4'), false);
});

test('injects an OpenAI-compatible function call and response pair', () => {
    const data = request();
    const result = injectFunctionResponsePrefill(data, DEFAULT_SETTINGS, 'ALPHA-', {
        idFactory: () => 'sfr_test',
    });

    assert.equal(result.injected, true);
    assert.equal(data.messages.length, 3);
    assert.deepEqual(data.messages[1], {
        role: 'assistant',
        content: '',
        tool_calls: [{
            id: 'sfr_test',
            type: 'function',
            function: {
                name: TOOL_NAME,
                arguments: '{"mode":"assistant_prefill"}',
            },
        }],
    });
    assert.deepEqual(data.messages[2], {
        role: 'tool',
        name: TOOL_NAME,
        tool_call_id: 'sfr_test',
        content: 'ALPHA-',
    });
});

test('uses an empty string rather than null for proxy compatibility', () => {
    const data = request();
    injectFunctionResponsePrefill(data, DEFAULT_SETTINGS, 'prefix', { idFactory: () => 'sfr_test' });
    assert.equal(data.messages.at(-2).content, '');
});

test('preserves the prefill exactly, including trailing whitespace', () => {
    const data = request();
    const prefix = 'Line one\nLine two  ';
    injectFunctionResponsePrefill(data, DEFAULT_SETTINGS, prefix, { idFactory: () => 'sfr_test' });
    assert.equal(data.messages.at(-1).content, prefix);
});

test('does not inject into non-Gemini models by default', () => {
    const data = request({ model: 'gpt-5.4' });
    const result = injectFunctionResponsePrefill(data, DEFAULT_SETTINGS, 'prefix');
    assert.equal(result.injected, false);
    assert.equal(result.reason, 'not-gemini');
    assert.equal(data.messages.length, 1);
});

test('can explicitly inject into non-Gemini models', () => {
    const data = request({ model: 'custom-model' });
    const settings = { ...DEFAULT_SETTINGS, geminiOnly: false };
    const result = injectFunctionResponsePrefill(data, settings, 'prefix', { idFactory: () => 'sfr_test' });
    assert.equal(result.injected, true);
});

test('quiet, continue and impersonate generations are opt-in', () => {
    for (const type of ['quiet', 'continue', 'impersonate']) {
        const data = request({ type });
        const result = injectFunctionResponsePrefill(data, DEFAULT_SETTINGS, 'prefix');
        assert.equal(result.injected, false, type);
    }

    const data = request({ type: 'quiet' });
    const settings = { ...DEFAULT_SETTINGS, includeQuiet: true };
    assert.equal(injectFunctionResponsePrefill(data, settings, 'prefix').injected, true);
});

test('upgrades prompt post-processing to keep tool messages', () => {
    for (const [input, expected] of [
        ['claude', 'merge_tools'],
        ['merge', 'merge_tools'],
        ['semi', 'semi_tools'],
        ['strict', 'strict_tools'],
    ]) {
        const data = request({ custom_prompt_post_processing: input });
        const result = preserveToolMessages(data);
        assert.equal(result.ok, true);
        assert.equal(data.custom_prompt_post_processing, expected);
    }
});

test('refuses single-user-message post-processing because it destroys the pair', () => {
    const data = request({ custom_prompt_post_processing: 'single' });
    const result = injectFunctionResponsePrefill(data, DEFAULT_SETTINGS, 'prefix');
    assert.equal(result.injected, false);
    assert.match(result.reason, /removes function calls/);
    assert.equal(data.messages.length, 1);
});

test('replaces an existing synthetic tail instead of duplicating it', () => {
    const data = request();
    injectFunctionResponsePrefill(data, DEFAULT_SETTINGS, 'first', { idFactory: () => 'sfr_first' });
    injectFunctionResponsePrefill(data, DEFAULT_SETTINGS, 'second', { idFactory: () => 'sfr_second' });

    assert.equal(data.messages.length, 3);
    assert.equal(data.messages.at(-2).tool_calls[0].id, 'sfr_second');
    assert.equal(data.messages.at(-1).content, 'second');
});

test('does not crash on an unrelated empty assistant message', () => {
    const data = request({
        messages: [
            { role: 'user', content: 'Hello' },
            { role: 'assistant', content: '' },
        ],
    });

    const result = injectFunctionResponsePrefill(data, DEFAULT_SETTINGS, 'prefix', {
        idFactory: () => 'sfr_test',
    });

    assert.equal(result.injected, true);
    assert.equal(data.messages.length, 4);
});

test('does not mutate existing tool declarations', () => {
    const tools = [{ type: 'function', function: { name: 'weather' } }];
    const data = request({ tools });
    injectFunctionResponsePrefill(data, DEFAULT_SETTINGS, 'prefix', { idFactory: () => 'sfr_test' });
    assert.equal(data.tools, tools);
    assert.deepEqual(data.tools, [{ type: 'function', function: { name: 'weather' } }]);
});

test('removes only an exact trailing Start Reply With assistant message', () => {
    const messages = [
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'ALPHA-' },
    ];

    assert.equal(removeTrailingAssistantPrefill(messages, 'ALPHA-'), true);
    assert.deepEqual(messages, [{ role: 'user', content: 'Hello' }]);
});

test('preserves ordinary history when Start Reply With is not trailing', () => {
    const messages = [
        { role: 'assistant', content: 'ALPHA-' },
        { role: 'user', content: 'Continue' },
    ];

    assert.equal(removeTrailingAssistantPrefill(messages, 'ALPHA-'), false);
    assert.equal(messages.length, 2);
});

test('native Start Reply With mode replaces direct assistant prefill with function response', () => {
    const data = request({
        messages: [
            { role: 'user', content: 'Complete ALPHA-BETA.' },
            { role: 'assistant', content: 'ALPHA-' },
        ],
    });
    const settings = { ...DEFAULT_SETTINGS, useStartReplyWith: true };
    const result = injectFunctionResponsePrefill(data, settings, 'ALPHA-', {
        idFactory: () => 'sfr_native_test',
    });

    assert.equal(result.injected, true);
    assert.equal(result.nativePrefillRemoved, true);
    assert.equal(data.messages.length, 3);
    assert.equal(data.messages.at(-2).content, '');
    assert.equal(data.messages.at(-1).content, 'ALPHA-');
    assert.equal(data.messages.some(message => message.role === 'assistant' && message.content === 'ALPHA-'), false);
});

test('normalizes native proxy URLs to their OpenAI-compatible API base', () => {
    assert.equal(getOpenAIProxyBase('https://example.test/proxy'), 'https://example.test/proxy/v1');
    assert.equal(getOpenAIProxyBase('https://example.test/proxy/'), 'https://example.test/proxy/v1');
    assert.equal(getOpenAIProxyBase('https://example.test/proxy/v1'), 'https://example.test/proxy/v1');
    assert.equal(getOpenAIProxyBase('https://example.test/proxy/v1beta'), 'https://example.test/proxy/v1');
});

test('routes native Gemini reverse proxies through their OpenAI-compatible endpoint', () => {
    for (const source of ['makersuite', 'vertexai']) {
        const data = request({
            chat_completion_source: source,
            reverse_proxy: 'https://example.test/proxy/gemini',
            proxy_password: 'secret',
        });
        const result = routeNativeGeminiProxyThroughOpenAI(data, true);

        assert.equal(result.routed, true);
        assert.equal(result.from, source);
        assert.equal(data.chat_completion_source, 'openai');
        assert.equal(data.reverse_proxy, 'https://example.test/proxy/gemini/v1');
        assert.equal(data.proxy_password, 'secret');
    }
});

test('leaves direct native Gemini and unrelated sources untouched', () => {
    const direct = request({ chat_completion_source: 'makersuite' });
    const custom = request({
        chat_completion_source: 'custom',
        reverse_proxy: 'https://example.test/proxy',
    });

    assert.deepEqual(routeNativeGeminiProxyThroughOpenAI(direct, true), { routed: false });
    assert.deepEqual(routeNativeGeminiProxyThroughOpenAI(custom, true), { routed: false });
    assert.equal(direct.chat_completion_source, 'makersuite');
    assert.equal(custom.chat_completion_source, 'custom');
});

test('proxy routing is disabled by default', () => {
    const data = request({
        chat_completion_source: 'makersuite',
        reverse_proxy: 'https://example.test/proxy/gemini',
        proxy_password: 'secret',
    });
    const result = injectFunctionResponsePrefill(data, DEFAULT_SETTINGS, 'ALPHA-', {
        idFactory: () => 'sfr_native_proxy_disabled',
    });

    assert.equal(result.injected, true);
    assert.equal(result.transport.routed, false);
    assert.equal(data.chat_completion_source, 'makersuite');
    assert.equal(data.reverse_proxy, 'https://example.test/proxy/gemini');
});

test('injection reroutes a native Gemini reverse proxy when enabled', () => {
    const data = request({
        chat_completion_source: 'makersuite',
        reverse_proxy: 'https://example.test/proxy/gemini',
        proxy_password: 'secret',
    });
    const settings = { ...DEFAULT_SETTINGS, useOpenAIProxyTransport: true };
    const result = injectFunctionResponsePrefill(data, settings, 'ALPHA-', {
        idFactory: () => 'sfr_native_proxy',
    });

    assert.equal(result.injected, true);
    assert.equal(result.transport.routed, true);
    assert.equal(data.chat_completion_source, 'openai');
    assert.equal(data.reverse_proxy, 'https://example.test/proxy/gemini/v1');
});
