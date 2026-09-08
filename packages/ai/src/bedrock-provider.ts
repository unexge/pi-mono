import { stream, streamSimple } from "./api/bedrock-converse-stream.ts";
import {
	stream as streamBedrockOpenAIResponses,
	streamSimple as streamSimpleBedrockOpenAIResponses,
} from "./api/bedrock-openai-responses.ts";

export const bedrockProviderModule = {
	stream,
	streamSimple,
};

export const bedrockOpenAIResponsesProviderModule = {
	stream: streamBedrockOpenAIResponses,
	streamSimple: streamSimpleBedrockOpenAIResponses,
};
