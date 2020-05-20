#pragma once

#include <torch/csrc/distributed/rpc/message.h>
#include <torch/csrc/distributed/rpc/rpc_agent.h>
#include <torch/csrc/distributed/rpc/rpc_command_base.h>

namespace torch {
namespace distributed {
namespace autograd {
class TORCH_API RpcWithProfilingResp : public rpc::RpcCommandBase {
 public:
  // For sending RPCs over the wire
  RpcWithProfilingResp(
      rpc::worker_id_t fromWorkerId,
      rpc::MessageType messageType,
      rpc::Message&& wrappedMessage,
      std::string profiledEvents,
      std::string profilingKey);

  // For receving RPCs. Used in from message when converting a message received
  // over the wire.
  RpcWithProfilingResp(
      rpc::worker_id_t fromWorkerId,
      rpc::MessageType messageType,
      std::unique_ptr<rpc::RpcCommandBase> wrappedRpc,
      rpc::MessageType wrappedMessageType,
      std::vector<torch::Tensor> tensors,
      std::string profiledEvents,
      std::string profilingKey);
  rpc::Message toMessageImpl() && override;
  static std::unique_ptr<RpcWithProfilingResp> fromMessage(
      const rpc::Message& message);
  // Retrieve the profiled events string
  std::string getProfiledEvents() const;
  // Retrieve profiling key
  std::string getProfilingKey() const;
  // Retrieve the original RPC which this ProfilingRPC wraps.
  RpcCommandBase& wrappedRpc();
  // Destructively move the wrapped RPC.
  std::unique_ptr<RpcCommandBase> moveWrappedRpc() &&;
  // Message type of the wrapped RPC
  rpc::MessageType wrappedMessageType() const;
  // retrieve the WID from which the rpc came from
  rpc::worker_id_t fromWorkerId() const;
  void setWrappedRpc(std::unique_ptr<RpcCommandBase> wrappedRpc);

 private:
  // the worker id
  rpc::worker_id_t fromWorkerId_;
  // message type
  rpc::MessageType messageType_;
  // wrapped message
  rpc::Message wrappedMessage_;
  std::unique_ptr<RpcCommandBase> wrappedRpc_;
  rpc::MessageType wrappedMessageType_;
  std::vector<torch::Tensor> tensors_;
  std::string profiledEvents_;
  std::string profilingKey_;
};
} // namespace autograd
} // namespace distributed
} // namespace torch
