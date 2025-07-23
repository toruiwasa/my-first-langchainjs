import { BedrockEmbeddings } from "@langchain/aws";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

export const semanticSerachSample = async (filePath: string) => {
  const loader = new PDFLoader(filePath);

  const docs = await loader.load();
  console.log(docs.length);

  console.log(docs[0].pageContent.slice(0, 200));
  console.log(docs[0].metadata);

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const allSplits = await textSplitter.splitDocuments(docs);
  console.log(allSplits.length);
  console.log(allSplits[0]);

  const embeddings = new BedrockEmbeddings({
    region: process.env.BEDROCK_AWS_REGION,
    model: "amazon.titan-embed-text-v2:0",
  });

  const vector1 = await embeddings.embedQuery(allSplits[0].pageContent);
  const vector2 = await embeddings.embedQuery(allSplits[1].pageContent);

  console.assert(vector1.length === vector2.length);
  console.log(`Generated vectors of length ${String(vector1.length)}\n`);
  console.log(vector1.slice(0, 10));

  const vectorStore = new MemoryVectorStore(embeddings);

  await vectorStore.addDocuments(allSplits);

  const result1 = await vectorStore.similaritySearch(
    "When was Netflix incorporated?",
  );
  console.log("Similarity search result: ", result1[0]);

  const result2 = await vectorStore.similaritySearchWithScore(
    "What is Netflix's revenue in 2025?",
  );

  console.log("Simirality search with score result: ", result2);

  // Retrievers
  const retriever = vectorStore.asRetriever({
    searchType: "mmr",
    searchKwargs: {
      fetchK: 1,
    },
  });

  const retrieverBatchResult = await retriever.batch([
    "When was Netflix incorporated?",
    "What was NIke's revenue in 2025?",
  ]);

  console.log("Retriever batch result: ", retrieverBatchResult);
};
