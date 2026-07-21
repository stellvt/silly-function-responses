export const MODULE_NAME = 'silly_function_responses';
export const TOOL_NAME = 'silly_prefill';

export const DEFAULT_SETTINGS = Object.freeze({
    enabled: true,
    prefill: '',
    useStartReplyWith: false,
    captureTrailingAssistantPrefill: false,
    geminiOnly: true,
    useOpenAIProxyTransport: false,
    includeQuiet: false,
    includeContinue: false,
    includeImpersonate: false,
});

const POST_PROCESSING_UPGRADES = Object.freeze({
    claude: 'merge_tools',
    merge: 'merge_tools',
    semi: 'semi_tools',
    strict: 'strict_tools',
});

const NATIVE_GEMINI_SOURCES = Object.freeze(['makersuite', 'vertexai']);

let callSequence = 0;

/**
 * Merge saved values with current defaults without mutating the input.
 * @param {object?} saved Previously saved settings.
 * @returns {object} Complete settings object.
 */
export function mergeSettings(saved) {
    return {
        ...DEFAULT_SETTINGS,
        ...(saved && typeof saved === 'object' ? saved : {}),
    };
}

/**
 * Check whether a model identifier belongs to Gemini.
 * @param {unknown} model Model identifier.
 * @returns {boolean} True for Gemini model identifiers.
 */
export function isGeminiModel(model) {
    return String(model ?? '').toLowerCase().includes('gemini');
}

/**
 * Check whether this generation type is enabled in extension settings.
 * @param {unknown} type SillyTavern generation type.
 * @param {object} settings Extension settings.
 * @returns {boolean} True when the type should receive a prefill.
 */
export function isGenerationTypeEnabled(type, settings) {
    switch (String(type ?? '').toLowerCase()) {
        case 'quiet':
            return Boolean(settings.includeQuiet);
        case 'continue':
            return Boolean(settings.includeContinue);
        case 'impersonate':
            return Boolean(settings.includeImpersonate);
        default:
            return true;
    }
}

/**
 * Upgrade SillyTavern prompt post-processing to a tool-preserving variant.
 * @param {object} generateData Final SillyTavern Chat Completion request object.
 * @returns {{ok: boolean, changedFrom?: string, changedTo?: string, reason?: string}}
 */
export function preserveToolMessages(generateData) {
    const current = String(generateData?.custom_prompt_post_processing ?? '');

    if (current === 'single') {
        return {
            ok: false,
            reason: 'single-user-message post-processing removes function calls',
        };
    }

    const upgraded = POST_PROCESSING_UPGRADES[current];
    if (upgraded) {
        generateData.custom_prompt_post_processing = upgraded;
        return { ok: true, changedFrom: current, changedTo: upgraded };
    }

    return { ok: true };
}

function isSyntheticTail(messages) {
    if (messages.length < 2) {
        return false;
    }

    const assistant = messages[messages.length - 2];
    const tool = messages[messages.length - 1];
    const call = Array.isArray(assistant?.tool_calls) ? assistant.tool_calls[0] : null;

    return assistant?.role === 'assistant'
        && assistant?.content === ''
        && Array.isArray(assistant.tool_calls)
        && assistant.tool_calls.length === 1
        && call?.function?.name === TOOL_NAME
        && typeof call?.id === 'string'
        && call.id.startsWith('sfr_')
        && tool?.role === 'tool'
        && tool?.name === TOOL_NAME
        && tool?.tool_call_id === call.id;
}

function nextCallId() {
    callSequence = (callSequence + 1) % Number.MAX_SAFE_INTEGER;
    return `sfr_${Date.now().toString(36)}_${callSequence.toString(36)}`;
}

/**
 * Build the OpenAI-compatible base URL exposed alongside a native Gemini proxy.
 * @param {string} reverseProxy Native Gemini reverse proxy URL.
 * @returns {string} OpenAI-compatible API base URL.
 */
export function getOpenAIProxyBase(reverseProxy) {
    const value = String(reverseProxy ?? '').trim().replace(/\/+$/, '');
    if (/\/v1beta$/i.test(value)) {
        return value.replace(/\/v1beta$/i, '/v1');
    }
    if (/\/v1$/i.test(value)) {
        return value;
    }
    return `${value}/v1`;
}

/**
 * Native Gemini proxy endpoints may reject functionCall/functionResponse parts
 * even when the same proxy supports their OpenAI tool-call representation. Keep
 * the SillyTavern connection native, but route this single generated request
 * through the proxy's OpenAI-compatible endpoint.
 *
 * Direct Google API requests are left untouched because their API key is held
 * by the SillyTavern server rather than included in generateData.
 *
 * @param {object} generateData Final SillyTavern Chat Completion request object.
 * @param {boolean} enabled Whether proxy transport rewriting is enabled.
 * @returns {{routed: boolean, from?: string, to?: string, reverseProxy?: string}}
 */
export function routeNativeGeminiProxyThroughOpenAI(generateData, enabled = false) {
    const source = String(generateData?.chat_completion_source ?? '').toLowerCase();
    const reverseProxy = String(generateData?.reverse_proxy ?? '').trim();

    if (!enabled || !NATIVE_GEMINI_SOURCES.includes(source) || !reverseProxy) {
        return { routed: false };
    }

    generateData.chat_completion_source = 'openai';
    generateData.reverse_proxy = getOpenAIProxyBase(reverseProxy);
    delete generateData.frequency_penalty;
    delete generateData.presence_penalty;

    return {
        routed: true,
        from: source,
        to: 'openai',
        reverseProxy: generateData.reverse_proxy,
    };
}

/**
 * Remove SillyTavern's trailing assistant message created by Start Reply With.
 * Only an exact trailing match is removed, so ordinary chat history is preserved.
 * @param {object[]} messages Final Chat Completion messages.
 * @param {string} prefill Resolved Start Reply With value.
 * @returns {boolean} True when a matching assistant prefill was removed.
 */
export function removeTrailingAssistantPrefill(messages, prefill) {
    if (!Array.isArray(messages) || !prefill || messages.length === 0) {
        return false;
    }

    const message = messages[messages.length - 1];
    const hasToolCalls = Array.isArray(message?.tool_calls) && message.tool_calls.length > 0;
    if (message?.role !== 'assistant' || hasToolCalls || message.content !== prefill) {
        return false;
    }

    messages.pop();
    return true;
}

/**
 * Read a plain trailing assistant message that can be treated as a prefill.
 * Tool calls and multimodal content are intentionally ignored.
 * @param {object[]} messages Final Chat Completion messages.
 * @returns {string|null} Exact prefill text, or null when no candidate exists.
 */
export function getTrailingAssistantPrefill(messages) {
    if (!Array.isArray(messages) || messages.length === 0) {
        return null;
    }

    const message = messages[messages.length - 1];
    const hasToolCalls = Array.isArray(message?.tool_calls) && message.tool_calls.length > 0;
    if (message?.role !== 'assistant' || hasToolCalls || typeof message.content !== 'string') {
        return null;
    }

    return message.content.length > 0 ? message.content : null;
}

/**
 * Append a synthetic OpenAI tool-call/result pair containing an assistant prefill.
 * SillyTavern converts this pair to Gemini functionCall/functionResponse for native
 * Google sources. OpenAI-compatible Gemini proxies can transform the same pair.
 *
 * @param {object} generateData Final SillyTavern Chat Completion request object.
 * @param {object} rawSettings Extension settings.
 * @param {unknown} resolvedPrefill Prefill after SillyTavern macro substitution.
 * @param {{idFactory?: () => string}} [options] Test/customization hooks.
 * @returns {{injected: boolean, reason?: string, callId?: string, model?: string, type?: string, prefillLength?: number, prefillSource?: string, postProcessing?: object, nativePrefillRemoved?: boolean, trailingPrefillRemoved?: boolean, transport?: object}}
 */
export function injectFunctionResponsePrefill(generateData, rawSettings, resolvedPrefill, options = {}) {
    const settings = mergeSettings(rawSettings);
    const model = String(generateData?.model ?? '');
    const type = String(generateData?.type ?? 'normal');

    if (!settings.enabled) {
        return { injected: false, reason: 'disabled', model, type };
    }

    if (!Array.isArray(generateData?.messages)) {
        return { injected: false, reason: 'messages-not-array', model, type };
    }

    if (settings.geminiOnly && !isGeminiModel(model)) {
        return { injected: false, reason: 'not-gemini', model, type };
    }

    if (!isGenerationTypeEnabled(type, settings)) {
        return { injected: false, reason: `generation-type-${type}-disabled`, model, type };
    }

    if (isSyntheticTail(generateData.messages)) {
        generateData.messages.splice(-2, 2);
    }

    const trailingPrefill = settings.captureTrailingAssistantPrefill
        ? getTrailingAssistantPrefill(generateData.messages)
        : null;
    const prefill = trailingPrefill ?? String(resolvedPrefill ?? '');
    const prefillSource = trailingPrefill !== null
        ? 'trailing-assistant'
        : settings.useStartReplyWith
            ? 'start-reply-with'
            : 'custom';

    if (prefill.length === 0) {
        return { injected: false, reason: 'empty-prefill', model, type };
    }

    const postProcessing = preserveToolMessages(generateData);
    if (!postProcessing.ok) {
        return { injected: false, reason: postProcessing.reason, model, type, postProcessing };
    }

    const trailingPrefillRemoved = trailingPrefill !== null
        ? removeTrailingAssistantPrefill(generateData.messages, trailingPrefill)
        : false;
    const nativePrefillRemoved = !trailingPrefillRemoved && settings.useStartReplyWith
        ? removeTrailingAssistantPrefill(generateData.messages, prefill)
        : false;

    const callId = (options.idFactory ?? nextCallId)();
    generateData.messages.push(
        {
            role: 'assistant',
            // This proxy rejects null here. An empty string is valid for both the
            // proxy and SillyTavern's Gemini converter.
            content: '',
            tool_calls: [
                {
                    id: callId,
                    type: 'function',
                    function: {
                        name: TOOL_NAME,
                        arguments: '{"mode":"assistant_prefill"}',
                    },
                },
            ],
        },
        {
            role: 'tool',
            name: TOOL_NAME,
            tool_call_id: callId,
            content: prefill,
        },
    );

    const transport = routeNativeGeminiProxyThroughOpenAI(
        generateData,
        settings.useOpenAIProxyTransport,
    );

    return {
        injected: true,
        callId,
        model,
        type,
        prefillLength: prefill.length,
        prefillSource,
        postProcessing,
        nativePrefillRemoved,
        trailingPrefillRemoved,
        transport,
    };
}
