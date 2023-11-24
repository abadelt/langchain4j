package dev.langchain4j.model.llamacpp;

import de.kherud.llama.InferenceParameters;
import de.kherud.llama.LlamaModel;
import de.kherud.llama.ModelParameters;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ChatMessageType;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.output.Response;
import lombok.Builder;
import lombok.extern.slf4j.Slf4j;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static java.util.Collections.singletonList;

@Slf4j
public class LlamaCppChatModel implements ChatLanguageModel {

    private final LlamaModel model;
    private final String modelPath;
    private final InferenceParameters inferParams;
    private final Double temperature;
    private final Double topP;
    private final Integer maxTokens;
    private final boolean logRequests;
    private final boolean logResponses;

    public final static String BOP = "[INST]";
    public final static String EOP = "[/INST]";

    @Builder
    public LlamaCppChatModel(String modelPath,
                             Double temperature,
                             Double topP,
                             Integer maxTokens,
                             Boolean logRequests,
                             Boolean logResponses) {
        this.temperature = temperature == null ? 0.7 : temperature;
        this.topP = topP;
        this.maxTokens = maxTokens;
        this.modelPath = modelPath;
        this.logResponses = logResponses;
        this.logRequests = logRequests;

        LlamaModel.setLogger(null); // (level, message) -> System.out.print(message));
        ModelParameters modelParams = new ModelParameters()
                .setNGpuLayers(43);
        inferParams = new InferenceParameters()
                .setTemperature(temperature.floatValue())
                .setPenalizeNl(true)
                .setMirostat(InferenceParameters.MiroStat.V2);
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        model = new LlamaModel(modelPath, modelParams);
    }

    @Override
    public Response<AiMessage> generate(List<ChatMessage> messages) {
        return generate(messages, null, null);
    }

    @Override
    public Response<AiMessage> generate(List<ChatMessage> messages, List<ToolSpecification> toolSpecifications) {
        return generate(messages, toolSpecifications, null);
    }

    @Override
    public Response<AiMessage> generate(List<ChatMessage> messages, ToolSpecification toolSpecification) {
        return generate(messages, singletonList(toolSpecification), toolSpecification);
    }

    private Response<AiMessage> generate(List<ChatMessage> messages,
                                         List<ToolSpecification> toolSpecifications,
                                         ToolSpecification toolThatMustBeExecuted) {
        String prompt = toLlamaCppPrompt(messages);

        log.debug("Calling Llama.cpp model.generate with prompt: " + prompt);

        if (logRequests) {
            log.debug("Prompt: " + prompt);
        }
        // String response = callJavaLLamaCpp(prompt);
        String response = callLlamaCppCLI(prompt);
        if (logResponses) {
            log.debug("Response: " + response);
        }
        log.debug("Llama.cpp model.generate response is: " + response.toString());

        AiMessage aiMessage = new AiMessage(response);

        return Response.from(aiMessage);
    }

    private String callJavaLLamaCpp(String prompt) {
        StringBuffer response = new StringBuffer();
        Iterable<LlamaModel.Output> outputIter = model.generate(prompt, inferParams);

        for (LlamaModel.Output output : outputIter) {
            log.debug(output.text);
            response.append(output.text);
        }
        return response.toString();
    }

    public static String toLlamaCppPrompt(List<ChatMessage> messages) {
        String prompt = new String("[INST]");
        prompt = messages.stream()
                .map(msg ->
                        ChatMessageType.SYSTEM.equals(msg.type()) ? "<<SYS>>" + msg.text() + "<</SYS>>" : msg.text())
                .reduce(prompt, String::concat);
        return prompt + ("[/INST]");
    }

    private String callLlamaCppCLI(String prompt) {
        try {
            prompt = prompt.replace("\n", "\\\n");
            prompt = prompt.replace("`", "\\`");

            String command = "./main -m " + this.modelPath + " -n -1 -c 4096 -b 4096 --n-gpu-layers 12 --prompt \"" + prompt + "\"";

            // Create ProcessBuilder
            ProcessBuilder processBuilder = new ProcessBuilder();

            // Set the command and any arguments
            processBuilder.directory(new File("/Users/andreas/Projekte/Playground/ai/llama.cpp"));
            processBuilder.command("bash", "-c", command); // For Unix-like systems

            // Start the process
            Process process = processBuilder.start();

            StringBuffer responseBuffer = new StringBuffer();
            // Read the output from the process
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                log.debug(line);
                responseBuffer.append(line + "\n");
            }
            String response = responseBuffer.toString();
            int endOfPromptIndex = response.lastIndexOf(EOP) + EOP.length();
            if (endOfPromptIndex > EOP.length()) {
                response = response.substring(endOfPromptIndex);
            }

            // Wait for the process to complete
            int exitCode = process.waitFor();
            log.debug("callLlamaCppCLI done with exit code: " + exitCode);

            return response.toString();
        } catch (IOException | InterruptedException e) {
            log.error("An exception occured when trying to call LLama.cpp CLI: ", e);
            return "";
        }
    }
}
