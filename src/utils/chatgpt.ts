import OpenAI from "openai";
import { ChatCompletionMessageParam } from "openai/resources";
import { logDebug } from "src/logDebug";
import { GoogleGenerativeAI, Part, Content } from "@google/generative-ai";

export type Message = {
	role: string;
	content: string;
};

export const streamResponse = async (
	apiKey: string,
	geminiApiKey: string,
	// prompt: string,
	messages: ChatCompletionMessageParam[],
	{
		max_tokens,
		model,
		temperature,
	}: {
		max_tokens?: number;
		model?: string;
		temperature?: number;
	} = {},
	cb: any
) => {
	logDebug("Calling AI :", {
		messages,
		model,
		max_tokens,
		temperature,
		isJSON: false,
	});

	if (model?.includes("gemini")) {
		const genAI = new GoogleGenerativeAI(geminiApiKey);
		const gemini = genAI.getGenerativeModel({ model });

		const systemPrompt = messages.find((m) => m.role === "system")?.content as string;
		const contents: Content[] = messages
			.filter((m) => m.role !== "system")
			.map((m) => ({ role: m.role === "assistant" ? "model" : "user", parts: [{ text: m.content as string }] }));

		const result = await gemini.generateContentStream({
			contents,
			systemInstruction: systemPrompt ? { role: "system", parts: [{text: systemPrompt}] } : undefined,
		});

		for await (const chunk of result.stream) {
			const chunkText = chunk.text();
			logDebug("AI chunk", { chunkText });
			cb(chunkText);
		}
		cb(null);
	} else {
		const openai = new OpenAI({
			apiKey: apiKey,
			dangerouslyAllowBrowser: true,
		});

		const stream = await openai.chat.completions.create({
			model: model || "gpt-4",
			messages,
			stream: true,
			max_tokens,
			temperature,
		});
		for await (const chunk of stream) {
			logDebug("AI chunk", { chunk });
			cb(chunk.choices[0]?.delta?.content || "");
		}
		cb(null);
	}
};

export const getResponse = async (
	apiKey: string,
	geminiApiKey: string,
	// prompt: string,
	messages: ChatCompletionMessageParam[],
	{
		model,
		max_tokens,
		temperature,
		isJSON,
	}: {
		model?: string;
		max_tokens?: number;
		temperature?: number;
		isJSON?: boolean;
	} = {}
) => {
	logDebug("Calling AI :", {
		messages,
		model,
		max_tokens,
		temperature,
		isJSON,
	});

	if (model?.includes("gemini")) {
		const genAI = new GoogleGenerativeAI(geminiApiKey);
		const gemini = genAI.getGenerativeModel({ model });

		const systemPrompt = messages.find((m) => m.role === "system")?.content as string;
		const contents: Content[] = messages
			.filter((m) => m.role !== "system")
			.map((m) => ({ role: m.role === "assistant" ? "model" : "user", parts: [{ text: m.content as string }] }));

		const result = await gemini.generateContent({
			contents,
			systemInstruction: systemPrompt ? { role: "system", parts: [{text: systemPrompt}] } : undefined,
		});

		const response = await result.response;
		const text = response.text();

		logDebug("AI response", { text });
		return isJSON ? JSON.parse(text) : text;
	} else {
		const openai = new OpenAI({
			apiKey: apiKey,
			dangerouslyAllowBrowser: true,
		});

		const completion = await openai.chat.completions.create({
			model: model || "gpt-4-1106-preview",
			messages,
			max_tokens,
			temperature,
			response_format: { type: isJSON ? "json_object" : "text" },
		});

		logDebug("AI response", { completion });
		const content = completion.choices[0].message?.content;
		if (!content) {
			throw new Error("No content in response");
		}
		return isJSON ? JSON.parse(content) : content;
	}
};

let count = 0;
export const createImage = async (
	apiKey: string,
	prompt: string,
	{
		isVertical = false,
		model,
	}: {
		isVertical?: boolean;
		model?: string;
	}
) => {
	logDebug("Calling AI :", {
		prompt,
		model,
	});
	const openai = new OpenAI({
		apiKey: apiKey,
		dangerouslyAllowBrowser: true,
	});

	count++;
	const response = await openai.images.generate({
		model: model || "dall-e-3",
		prompt,
		n: 1,
		size: isVertical ? "1024x1792" : "1792x1024",
		response_format: "b64_json",
	});
	logDebug("AI response", { response });
	if (!response.data || !response.data[0]) {
		throw new Error("No image data returned from API");
	}
	const b64_json = response.data[0].b64_json;
	if (!b64_json) {
		throw new Error("No image data returned from API");
	}
	return b64_json;
};
