import {
    DEFAULT_SETTINGS,
    MODULE_NAME,
    injectFunctionResponsePrefill,
    mergeSettings,
} from './lib.js';

const EXTENSION_PATH = 'third-party/silly-function-responses';
const STATUS_ID = 'sfr_status';

const context = SillyTavern.getContext();
const {
    eventSource,
    eventTypes,
    extensionSettings,
    powerUserSettings,
    renderExtensionTemplateAsync,
    saveSettingsDebounced,
    substituteParams,
} = context;

let latestStatus = {
    kind: 'idle',
    text: 'Waiting for generation.',
};
let renderSettingsPromise = null;

function getSettings() {
    const previous = extensionSettings[MODULE_NAME];
    const merged = mergeSettings(previous);
    extensionSettings[MODULE_NAME] = merged;

    if (!previous || Object.keys(DEFAULT_SETTINGS).some(key => !Object.hasOwn(previous, key))) {
        saveSettingsDebounced();
    }

    return merged;
}

function setStatus(kind, text) {
    latestStatus = { kind, text };
    const status = document.getElementById(STATUS_ID);
    if (!status) {
        return;
    }

    status.dataset.kind = kind;
    status.textContent = text;
}

function updateCharacterCount() {
    const settings = getSettings();
    const source = settings.useStartReplyWith
        ? powerUserSettings.user_prompt_bias
        : settings.prefill;
    const counter = document.getElementById('sfr_prefill_count');
    if (counter) {
        counter.textContent = `${String(source ?? '').length} chars`;
    }
}

function updatePrefillSourceUi() {
    const settings = getSettings();
    const useNative = Boolean(settings.useStartReplyWith);
    const prefill = document.getElementById('sfr_prefill');
    const clearButton = document.getElementById('sfr_clear_prefill');

    if (prefill instanceof HTMLTextAreaElement) {
        prefill.disabled = useNative;
    }
    if (clearButton instanceof HTMLInputElement) {
        clearButton.disabled = useNative;
    }

    updateCharacterCount();
}

function bindCheckbox(id, key, onChange = null) {
    const element = document.getElementById(id);
    if (!(element instanceof HTMLInputElement)) {
        return;
    }

    const settings = getSettings();
    element.checked = Boolean(settings[key]);
    element.addEventListener('change', () => {
        getSettings()[key] = element.checked;
        saveSettingsDebounced();
        onChange?.();
    });
}

async function renderSettings() {
    if (document.getElementById('sfr_settings')) {
        return;
    }

    if (renderSettingsPromise) {
        return renderSettingsPromise;
    }

    renderSettingsPromise = (async () => {
        const container = document.getElementById('extensions_settings2');
        if (!container) {
            console.warn('[Silly Function Responses] Extensions settings container was not found.');
            return;
        }

        const html = await renderExtensionTemplateAsync(EXTENSION_PATH, 'settings');
        if (document.getElementById('sfr_settings')) {
            return;
        }
        container.insertAdjacentHTML('beforeend', html);

        bindCheckbox('sfr_enabled', 'enabled');
        bindCheckbox('sfr_use_start_reply_with', 'useStartReplyWith', updatePrefillSourceUi);
        bindCheckbox('sfr_capture_trailing_assistant_prefill', 'captureTrailingAssistantPrefill');
        bindCheckbox('sfr_gemini_only', 'geminiOnly');
        bindCheckbox('sfr_use_openai_proxy_transport', 'useOpenAIProxyTransport');
        bindCheckbox('sfr_include_quiet', 'includeQuiet');
        bindCheckbox('sfr_include_continue', 'includeContinue');
        bindCheckbox('sfr_include_impersonate', 'includeImpersonate');

        const prefill = document.getElementById('sfr_prefill');
        if (prefill instanceof HTMLTextAreaElement) {
            prefill.value = String(getSettings().prefill ?? '');
            prefill.addEventListener('input', () => {
                getSettings().prefill = prefill.value;
                saveSettingsDebounced();
                updateCharacterCount();
            });
        }

        const clearButton = document.getElementById('sfr_clear_prefill');
        clearButton?.addEventListener('click', () => {
            const settings = getSettings();
            settings.prefill = '';
            if (prefill instanceof HTMLTextAreaElement) {
                prefill.value = '';
            }
            saveSettingsDebounced();
            updateCharacterCount();
            setStatus('idle', 'Prefill cleared.');
        });

        document.getElementById('start_reply_with')?.addEventListener('input', updatePrefillSourceUi);
        document.getElementById('chat-show-reply-prefix-checkbox')?.addEventListener('change', updatePrefillSourceUi);

        updatePrefillSourceUi();
        setStatus(latestStatus.kind, latestStatus.text);
    })().finally(() => {
        renderSettingsPromise = null;
    });

    return renderSettingsPromise;
}

function handleChatCompletionSettingsReady(generateData) {
    const settings = getSettings();
    let resolvedPrefill = '';

    try {
        const rawPrefill = settings.useStartReplyWith
            ? powerUserSettings.user_prompt_bias
            : settings.prefill;
        resolvedPrefill = substituteParams(String(rawPrefill ?? ''));
    } catch (error) {
        console.error('[Silly Function Responses] Macro substitution failed.', error);
        setStatus('error', 'Failed to resolve macros in the prefill. See the browser console for details.');
        return;
    }

    const result = injectFunctionResponsePrefill(generateData, settings, resolvedPrefill);
    if (result.injected) {
        const sourceLabels = {
            'custom': 'custom prefill',
            'start-reply-with': 'Start Reply With',
            'trailing-assistant': 'trailing assistant message',
        };
        const sourceNote = `; source: ${sourceLabels[result.prefillSource] ?? result.prefillSource}`;
        setStatus(
            'success',
            `Prefill applied: ${result.model || 'unknown model'}, ${result.prefillLength} chars${sourceNote}`,
        );
        console.debug('[Silly Function Responses] Synthetic function response injected.', {
            model: result.model,
            type: result.type,
            prefillLength: result.prefillLength,
            prefillSource: result.prefillSource,
            nativePrefillRemoved: result.nativePrefillRemoved,
            trailingPrefillRemoved: result.trailingPrefillRemoved,
            postProcessing: result.postProcessing,
            transport: result.transport,
        });
        return;
    }

    if (result.reason === 'single-user-message post-processing removes function calls') {
        setStatus('error', 'Single user message mode is incompatible with prefill.');
    }
}

getSettings();
eventSource.makeLast(eventTypes.CHAT_COMPLETION_SETTINGS_READY, handleChatCompletionSettingsReady);
eventSource.on(eventTypes.APP_READY, () => {
    setTimeout(() => {
        // Re-apply after every extension has loaded so fake prefill conversion
        // is the final CHAT_COMPLETION_SETTINGS_READY transformation.
        eventSource.makeLast(eventTypes.CHAT_COMPLETION_SETTINGS_READY, handleChatCompletionSettingsReady);
        void renderSettings();
    }, 0);
});

if (document.readyState !== 'loading') {
    setTimeout(() => void renderSettings(), 0);
}
