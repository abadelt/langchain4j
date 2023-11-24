package dev.langchain4j.model.llamacpp;

import dev.langchain4j.model.chat.ChatLanguageModel;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class LlamaCppChatModelIT {

    @Test
    @Disabled("need to determine where to put test model")
    void should_send_user_message_and_return_answer() {

        ChatLanguageModel model = LlamaCppChatModel.builder()
                .modelPath("/Users/andreas/Projekte/Playground/ai/llama.cpp/models/ggml-vocab-llama.gguf")
                .maxTokens(3)
                .temperature(0.7)
                .logRequests(true)
                .logResponses(true)
                .build();

        String answer = model.generate("Say 'hello'");

        assertThat(answer).containsIgnoringCase("hello");
        System.out.println(answer);
    }
}