# Docker Core ↔ Native MuJoCo Protocol

## Transport

- WebSocket over localhost/Docker Desktop host networking.
- Default native endpoint: `ws://127.0.0.1:8765`.
- Docker client endpoint: `ws://host.docker.internal:8765`.
- Start with JSON control messages plus binary image payloads or MessagePack. Move to a different encoding only after profiling.
- Enforce message-size and rate limits.

## Envelope

Every message carries:

```json
{
  "protocol_version": "1.0",
  "type": "state",
  "sequence": 101,
  "wall_time_ns": 1786492800000000000,
  "sim_step": 5500,
  "sim_time_s": 11.0,
  "request_id": null,
  "payload": {}
}
```

`wall_time_ns` is diagnostic. Ordering and simulation behavior use `sequence`, `sim_step`, and `sim_time_s`.

## Handshake

Client sends:

- supported protocol versions;
- client name/version;
- requested features;
- expected robot model ID/hash;
- expected scene schema versions;
- maximum accepted image/message sizes.

Server replies:

- selected protocol version;
- server/MuJoCo versions;
- capabilities;
- model ID/hash;
- loaded scene revision/hash;
- joint/camera names;
- default and maximum rates/resolutions.

Reject incompatible major versions and model mismatches unless an explicit development override is enabled.

## Client-to-server messages

- `hello`
- `load_scene`
- `reset`
- `pause`
- `step` for lockstep/test mode
- `joint_command`
- `set_camera_config`
- `subscribe`
- `unsubscribe`
- `ping`
- `shutdown` only when locally authorized

## Server-to-client messages

- `hello_ack`
- `status`
- `state`
- `object_state`
- `camera_metadata`
- `camera_frame`
- `contact_event`
- `ack`
- `error`
- `pong`

## State message requirements

- coherent joint vector from one simulation step;
- ordered or named joints plus mapping hash;
- command sequence last applied;
- scene revision;
- paused/degraded state;
- optional tracked-object poses and contact summary.

## Camera frame requirements

- camera name (`left` or `right` initially);
- frame sequence;
- same `sim_step` for a stereo pair;
- width, height, format, encoded byte length;
- camera intrinsics/profile ID;
- scene revision and model hash;
- render start/end or duration metadata;
- optional depth/segmentation channel descriptors.

Do not put image bytes in unbounded JSON strings in production. Use binary WebSocket frames associated with a small metadata header.

## Flow control

- Physics/state and camera rates are independent.
- State updates may be coalesced to latest if the client falls behind.
- Camera stream is latest-frame-wins unless recording mode explicitly requests lossless behavior.
- Commands use bounded queues and return overload errors instead of growing without limit.
- Slow browser/SDK consumers must never stall the native simulation loop.

## Failure behavior

- Heartbeat interval and timeout are negotiated.
- On missed heartbeat, Docker backend becomes `DEGRADED` and marks snapshot age.
- gRPC operations have bounded deadlines/policy-defined errors.
- Reconnect performs a fresh handshake and model/scene hash check.
- After reconnect, command replay is opt-in; never blindly replay stale actuator commands.
- Reset/reload invalidates old frame and object revisions.

## Security boundary

- Listen on loopback by default.
- No arbitrary file paths supplied by remote client.
- Scene loading uses content or approved scene IDs, not unrestricted host filesystem access.
- No remote URLs in scene assets by default.
- Validate all numeric ranges and lengths before allocation.
- Do not implement arbitrary Python evaluation, shell commands, or plugin loading over the protocol.
