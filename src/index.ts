import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatBedrockConverse } from "@langchain/aws";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import path from "path";
import { semanticSerachSample } from "./samples/semanticSearchSample.js";

const model = new ChatBedrockConverse({
  model: "us.amazon.nova-micro-v1:0",
  region: process.env.BEDROCK_AWS_REGION ?? "us-east-1",
  credentials: {
    secretAccessKey: process.env.BEDROCK_AWS_SECRET_ACCESS_KEY ?? "",
    accessKeyId: process.env.BEDROCK_AWS_ACCESS_KEY_ID ?? "",
  },
  temperature: 0,
});

const directModelInvokingSample = async () => {
  const response = await model.invoke([
    new SystemMessage("You are a helpful assistant!"),
    new HumanMessage("Hello world!"),
  ]);
  console.log(response);
};
// await directModelInvokingSample();

const promptTemplateSample = async () => {
  const systemTemplate = "Translate the following from English into {language}";
  const promptTemplate = ChatPromptTemplate.fromMessages([
    ["system", systemTemplate],
    ["user", "{text}"],
  ]);
  const promptValue = await promptTemplate.invoke({
    language: "japanese",
    text: "hi!",
  });

  console.log(promptValue, promptValue.toChatMessages());

  const response = await model.invoke(promptValue);
  console.log(response.content);
};

// await promptTemplateSample();

await semanticSerachSample(
  path.join(import.meta.dirname, "../netflix_10q_2025.pdf"),
);
