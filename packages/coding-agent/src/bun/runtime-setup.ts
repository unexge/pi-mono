import { bedrockOpenAIResponsesProviderModule, bedrockProviderModule } from "@earendil-works/pi-ai/bedrock-provider";
import { registerBunOAuthFlows } from "@earendil-works/pi-ai/bun-oauth";
import { setBedrockOpenAIResponsesProviderModule, setBedrockProviderModule } from "@earendil-works/pi-ai/compat";
import { APP_NAME } from "../config.ts";

process.title = APP_NAME;
process.emitWarning = (() => {}) as typeof process.emitWarning;
registerBunOAuthFlows();
setBedrockProviderModule(bedrockProviderModule);
setBedrockOpenAIResponsesProviderModule(bedrockOpenAIResponsesProviderModule);
