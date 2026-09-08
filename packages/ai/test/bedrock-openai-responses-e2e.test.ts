import { Type } from "typebox";
import { describe, expect, it } from "vitest";
import { complete, getModel, stream } from "../src/compat.ts";
import type { Context, Tool } from "../src/types.ts";
import { hasBedrockCredentials } from "./bedrock-utils.ts";

const enabled = hasBedrockCredentials() && process.env.BEDROCK_OPENAI_RESPONSES_TEST === "1";
const model = getModel("amazon-bedrock", "global.openai.gpt-5.6-sol");

const calculatorParameters = Type.Object({
	a: Type.Number(),
	b: Type.Number(),
});

const calculator: Tool<typeof calculatorParameters> = {
	name: "add_numbers",
	description: "Add two numbers and return their sum",
	parameters: calculatorParameters,
};

describe.skipIf(!enabled)("Bedrock OpenAI Responses E2E", () => {
	it("streams a basic response", { timeout: 60_000 }, async () => {
		const context: Context = {
			messages: [{ role: "user", content: "Reply with exactly BEDROCK_RESPONSES_OK", timestamp: Date.now() }],
		};
		const responseStream = stream(model, context, { maxTokens: 32, cacheRetention: "none" });
		let text = "";
		for await (const event of responseStream) {
			if (event.type === "text_delta") text += event.delta;
		}
		const response = await responseStream.result();

		expect(response.stopReason, response.errorMessage).toBe("stop");
		expect(text).toContain("BEDROCK_RESPONSES_OK");
		expect(response.usage.input).toBeGreaterThan(0);
		expect(response.usage.output).toBeGreaterThan(0);
	});

	it("completes a tool-use round trip", { timeout: 120_000 }, async () => {
		const context: Context = {
			messages: [
				{
					role: "user",
					content: "Use add_numbers to calculate 19 + 23, then report the result.",
					timestamp: Date.now(),
				},
			],
			tools: [calculator],
		};
		const first = await complete(model, context, { maxTokens: 128, cacheRetention: "none" });
		expect(first.stopReason, first.errorMessage).toBe("toolUse");
		const toolCall = first.content.find((block) => block.type === "toolCall");
		if (!toolCall || toolCall.type !== "toolCall") throw new Error("Expected add_numbers tool call");
		expect(toolCall.name).toBe("add_numbers");
		expect(toolCall.arguments).toMatchObject({ a: 19, b: 23 });

		context.messages.push(first, {
			role: "toolResult",
			toolCallId: toolCall.id,
			toolName: toolCall.name,
			content: [{ type: "text", text: "42" }],
			isError: false,
			timestamp: Date.now(),
		});
		const second = await complete(model, context, { maxTokens: 64, cacheRetention: "none" });
		expect(second.stopReason, second.errorMessage).toBe("stop");
		expect(second.content.map((block) => (block.type === "text" ? block.text : "")).join("")).toContain("42");
	});

	it("writes and reuses a prompt cache entry", { timeout: 120_000 }, async () => {
		const stablePrompt = `${"Stable Bedrock Responses cache integration prefix. ".repeat(600)}\nReply with exactly CACHE_OK.`;
		const context: Context = {
			messages: [{ role: "user", content: stablePrompt, timestamp: Date.now() }],
		};
		const options = {
			maxTokens: 32,
			cacheRetention: "long" as const,
			sessionId: "pi-bedrock-responses-e2e-v1",
		};

		const first = await complete(model, context, options);
		expect(first.stopReason, first.errorMessage).toBe("stop");
		expect(first.usage.cacheWrite + first.usage.cacheRead).toBeGreaterThan(0);

		const second = await complete(model, context, options);
		expect(second.stopReason, second.errorMessage).toBe("stop");
		expect(second.usage.cacheRead).toBeGreaterThan(0);
	});
});
