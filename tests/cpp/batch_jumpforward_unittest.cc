#include <gtest/gtest.h>

#include "serve/engine_actions/action.h"

namespace mlc {
namespace llm {
namespace serve {

TEST(BatchJumpForward, PlainTextDoesNotAccessModelOrPreempt) {
  auto config = GenerationConfig(tvm::ffi::make_object<GenerationConfigNode>());
  Request request("plain-text", {TokenData(std::vector<int>{0})}, config);
  RequestStateEntry entry(request, 1, 0, 0, {}, std::nullopt);
  entry->status = RequestStateStatus::kAlive;
  entry->mstates[0]->inputs.clear();
  RequestState state({entry}, 1, std::chrono::high_resolution_clock::now());
  request->rstate = state.get();
  EngineState estate;
  estate->running_queue.push_back(request);

  // A null model makes any capacity query fail: ineligible requests must return before
  // querying available pages, regardless of how close the real cache is to capacity.
  auto action = EngineAction::BatchJumpForward({Model(nullptr)}, Tokenizer(nullptr), std::nullopt);
  EXPECT_TRUE(action->Step(estate).empty());
  EXPECT_EQ(entry->status, RequestStateStatus::kAlive);
  EXPECT_EQ(estate->running_queue.size(), 1);
  EXPECT_TRUE(estate->waiting_queue.empty());
  EXPECT_TRUE(entry->mstates[0]->inputs.empty());
}

}  // namespace serve
}  // namespace llm
}  // namespace mlc
