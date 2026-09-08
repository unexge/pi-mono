import { afterEach, describe, expect, it, vi } from "vitest";

const tokenMock = vi.hoisted(() => ({
	configs: [] as unknown[],
}));

vi.mock("@aws/bedrock-token-generator", () => ({
	getTokenProvider: (config: unknown) => {
		tokenMock.configs.push(config);
		return async () => "generated-token";
	},
}));

import {
	type BedrockOpenAIResponsesOptions,
	stream as streamBedrockOpenAIResponses,
} from "../src/api/bedrock-openai-responses.ts";
import { getBuiltinModel } from "../src/providers/all.ts";
import type { Context } from "../src/types.ts";

const model = getBuiltinModel("amazon-bedrock", "global.openai.gpt-5.6-sol");
const context: Context = {
	messages: [{ role: "user", content: "Reply with exactly OK.", timestamp: 1 }],
};

type CapturedRequest = {
	url: string;
	authorization: string | null;
	body: Record<string, unknown>;
};

function completedResponse(): Response {
	const event = {
		type: "response.completed",
		sequence_number: 0,
		response: {
			id: "resp_bedrock_test",
			status: "completed",
			output: [],
			usage: {
				input_tokens: 1,
				output_tokens: 1,
				total_tokens: 2,
				input_tokens_details: { cached_tokens: 0 },
			},
		},
	};
	return new Response(`data: ${JSON.stringify(event)}\n\ndata: [DONE]\n\n`, {
		status: 200,
		headers: { "content-type": "text/event-stream" },
	});
}

async function captureRequest(options: BedrockOpenAIResponsesOptions): Promise<CapturedRequest> {
	let captured: CapturedRequest | undefined;
	vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
		const request = new Request(input, init);
		captured = {
			url: request.url,
			authorization: request.headers.get("authorization"),
			body: JSON.parse(await request.text()) as Record<string, unknown>,
		};
		return completedResponse();
	});

	const response = await streamBedrockOpenAIResponses(model, context, options).result();
	expect(response.stopReason, response.errorMessage).toBe("stop");
	if (!captured) throw new Error("Request was not captured");
	return captured;
}

describe("Bedrock OpenAI Responses", () => {
	afterEach(() => {
		vi.restoreAllMocks();
		tokenMock.configs.length = 0;
	});

	it("uses Bedrock bearer auth and GPT-5.6 prompt caching fields", async () => {
		const request = await captureRequest({
			bearerToken: "bedrock-token",
			sessionId: "session-123",
			cacheRetention: "long",
		});

		expect(request.url).toBe("https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/responses");
		expect(request.authorization).toBe("Bearer bedrock-token");
		expect(request.body).toMatchObject({
			model: "global.openai.gpt-5.6-sol",
			prompt_cache_key: "session-123",
			prompt_cache_options: { ttl: "30m" },
		});
		expect(request.body).not.toHaveProperty("prompt_cache_retention");
		expect(tokenMock.configs).toHaveLength(0);
	});

	it("generates a short-term token from a scoped AWS profile", async () => {
		const request = await captureRequest({
			profile: "test-profile",
			region: "us-west-2",
			cacheRetention: "none",
			env: {
				AWS_ACCESS_KEY_ID: "ambient-access-key",
				AWS_SECRET_ACCESS_KEY: "ambient-secret-key",
			},
		});

		expect(tokenMock.configs).toEqual([{ region: "us-west-2", profile: "test-profile" }]);
		expect(request.url).toBe("https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1/responses");
		expect(request.authorization).toBe("Bearer generated-token");
		expect(request.body).toMatchObject({ prompt_cache_options: { mode: "explicit" } });
		expect(request.body).not.toHaveProperty("prompt_cache_key");
	});
});
