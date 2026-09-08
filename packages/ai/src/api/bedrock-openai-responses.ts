import { getTokenProvider } from "@aws/bedrock-token-generator";
import type { Model, ProviderEnv, ProviderHeaders, SimpleStreamOptions, StreamFunction } from "../types.ts";
import { getProviderEnvValue } from "../utils/provider-env.ts";
import {
	type OpenAIResponsesOptions,
	stream as openAIResponsesStream,
	streamSimple as openAIResponsesStreamSimple,
} from "./openai-responses.ts";

export interface BedrockOpenAIResponsesOptions extends OpenAIResponsesOptions {
	region?: string;
	profile?: string;
	bearerToken?: string;
}

const FALLBACK_REGION = "us-east-1";
const DUMMY_API_KEY = "aws-short-term-token";

function getRegionFromBaseUrl(baseUrl: string): string | undefined {
	try {
		return new URL(baseUrl).hostname.match(/^bedrock-runtime(?:-fips)?\.([a-z0-9-]+)\.amazonaws\.com(?:\.cn)?$/)?.[1];
	} catch {
		return undefined;
	}
}

function resolveRegion(model: Model<"openai-responses">, options: BedrockOpenAIResponsesOptions | undefined): string {
	return (
		options?.region ||
		getProviderEnvValue("AWS_REGION", options?.env) ||
		getProviderEnvValue("AWS_DEFAULT_REGION", options?.env) ||
		getRegionFromBaseUrl(model.baseUrl) ||
		FALLBACK_REGION
	);
}

function getStaticCredentials(env?: ProviderEnv) {
	const accessKeyId = getProviderEnvValue("AWS_ACCESS_KEY_ID", env);
	const secretAccessKey = getProviderEnvValue("AWS_SECRET_ACCESS_KEY", env);
	if (!accessKeyId || !secretAccessKey) return undefined;
	return {
		accessKeyId,
		secretAccessKey,
		sessionToken: getProviderEnvValue("AWS_SESSION_TOKEN", env),
	};
}

function getBearerToken(options: BedrockOpenAIResponsesOptions | undefined): string | undefined {
	return (
		options?.bearerToken ||
		options?.apiKey ||
		getProviderEnvValue("AWS_BEARER_TOKEN_BEDROCK", options?.env) ||
		undefined
	);
}

function createTokenFetch(options: BedrockOpenAIResponsesOptions | undefined, region: string): typeof fetch {
	const baseFetch = options?.fetch ?? globalThis.fetch;
	const profile = options?.profile || getProviderEnvValue("AWS_PROFILE", options?.env);
	const credentials = profile ? undefined : getStaticCredentials(options?.env);
	const provideToken = getTokenProvider({
		region,
		...(profile ? { profile } : {}),
		...(credentials ? { credentials } : {}),
	});

	return async (input, init) => {
		const request = new Request(input, init);
		const headers = new Headers(request.headers);
		headers.set("authorization", `Bearer ${await provideToken()}`);
		headers.delete("x-api-key");
		return baseFetch(request, { headers });
	};
}

function prepareOptions(
	model: Model<"openai-responses">,
	options: BedrockOpenAIResponsesOptions | undefined,
): { model: Model<"openai-responses">; options: OpenAIResponsesOptions } {
	const region = resolveRegion(model, options);
	const requestModel = {
		...model,
		baseUrl: `https://bedrock-runtime.${region}.amazonaws.com/openai/v1`,
	};
	const headers: ProviderHeaders = { ...model.headers, ...options?.headers };
	const bearerToken = getBearerToken(options);
	if (bearerToken) {
		return { model: requestModel, options: { ...options, apiKey: bearerToken, headers } };
	}
	return {
		model: requestModel,
		options: { ...options, apiKey: DUMMY_API_KEY, headers, fetch: createTokenFetch(options, region) },
	};
}

export const stream: StreamFunction<"openai-responses", BedrockOpenAIResponsesOptions> = (model, context, options) => {
	const prepared = prepareOptions(model, options);
	return openAIResponsesStream(prepared.model, context, prepared.options);
};

export const streamSimple: StreamFunction<"openai-responses", SimpleStreamOptions> = (model, context, options) => {
	const prepared = prepareOptions(model, options as BedrockOpenAIResponsesOptions | undefined);
	return openAIResponsesStreamSimple(prepared.model, context, prepared.options as SimpleStreamOptions);
};
