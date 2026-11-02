// pad_input.h - lets g1_ctrl take its gamepad from a laptop pad over UDP (Wi-Fi), alongside the Unitree remote.
//
// g1_ctrl reads its buttons and sticks from lowstate.wireless_remote (the Unitree remote) through
// UnitreeJoystick::extract(). FSMState::pre_run() merges the pad into those 40 raw bytes BEFORE extract(),
// so the pad behaves exactly like the remote: the same combos in config.yaml, the same on_pressed /
// pressed_time handling, the same velocity_commands observation. No config or policy changes.
//
// Rules (the pad is OFF unless g1_ctrl is started with --pad-port and --pad-key):
//   * The Unitree remote always keeps working. Buttons are OR-ed, so the spotter's L2 + B (Passive) wins.
//   * While the pad is FRESH (a valid packet less than --pad-timeout-ms ago) it owns the sticks.
//     Silent, disconnected or never seen: no pad buttons, and the sticks come from the remote again.
//   * Deadman: the pad's sticks only count while L1 is held (--pad-no-deadman for testing only).
//   * Packets need the shared key, come from --pad-allow-ip if given, and must have a rising sequence number
//     inside a live session (a restarted sender is accepted after 1 s of silence). Anything else is dropped
//     without an answer. UDP is not encrypted: the key keeps strangers and mistakes out, nothing more.
//   * F1 / F2 (used by software on the Jetson) are never set from the pad.
//
// Packet "GPDK" v1, little endian, 34 bytes (sender: g1-workspace/pad/pad_stream.py):
//   "GPDK" ver(u8) flags(u8, bit0 = pad connected) seq(u32) buttons(u32) axes(6 x i16, /32767) key(8)
//   buttons bits: A0 B1 X2 Y3 L1 4 R1 5 SELECT 6 START 7 MODE 8 L3 9 R3 10 DUP 11 DDOWN 12 DLEFT 13 DRIGHT 14 L2 15 R2 16
//   axes: LX LY RX RY L2 R2 (sticks -1..1, up = +1 on LY/RY, right = +1 on LX/RX; triggers 0..1)
#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#include <spdlog/spdlog.h>
#include <unitree/dds_wrapper/common/unitree_joystick.hpp>

class PadInput
{
public:
    struct Config
    {
        int port = 9872;
        uint8_t key[8] = {0};
        in_addr_t allow_ip = 0;     // 0 = any; the first valid packet pins the peer for the session
        double timeout_s = 0.25;    // silence before the pad counts as gone
        bool deadman = true;        // sticks only while L1 is held
    };

    static constexpr size_t kLen = 34;

    // logical button bits, same order as pad_proto.BUTTONS on the laptop
    enum Bit { A = 0, B = 1, X = 2, Y = 3, L1 = 4, R1 = 5, SELECT = 6, START = 7, MODE = 8, L3 = 9, R3 = 10,
               DUP = 11, DDOWN = 12, DLEFT = 13, DRIGHT = 14, L2 = 15, R2 = 16 };

    explicit PadInput(const Config& cfg) : cfg_(cfg)
    {
        fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (fd_ < 0) throw std::runtime_error("pad input: socket() failed");
        int one = 1;
        setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        sockaddr_in me{};
        me.sin_family = AF_INET;
        me.sin_addr.s_addr = htonl(INADDR_ANY);
        me.sin_port = htons(static_cast<uint16_t>(cfg_.port));
        if (::bind(fd_, reinterpret_cast<sockaddr*>(&me), sizeof me) != 0)
        {
            ::close(fd_);
            throw std::runtime_error("pad input: cannot bind UDP port " + std::to_string(cfg_.port));
        }
        run_ = true;
        thread_ = std::thread([this] { loop(); });
    }

    ~PadInput()
    {
        run_ = false;
        if (thread_.joinable()) thread_.join();
        if (fd_ >= 0) ::close(fd_);
    }

    PadInput(const PadInput&) = delete;
    PadInput& operator=(const PadInput&) = delete;

    // Call once per FSM tick, BEFORE joystick.extract(key). `key` holds the remote's raw bytes.
    void merge(unitree::common::REMOTE_DATA_RX& key)
    {
        Snapshot s;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            s = snap_;
        }
        const auto now = std::chrono::steady_clock::now();
        const bool fresh = s.have && s.connected &&
                           std::chrono::duration<double>(now - s.rx).count() < cfg_.timeout_s;
        if (fresh != was_fresh_)
        {
            was_fresh_ = fresh;
            if (fresh) spdlog::info("Pad: connected. L1 is the deadman for the sticks; the remote still works.");
            else       spdlog::warn("Pad: silent or disconnected: no pad buttons, sticks neutral. The remote still works.");
        }
        if (!fresh) return;

        auto bit = [&](int b) -> bool { return (s.buttons >> b) & 1u; };
        auto& c = key.RF_RX.btn.components;
        if (bit(R1)) c.R1 = 1;
        if (bit(L1)) c.L1 = 1;
        if (bit(START)) c.Start = 1;
        if (bit(SELECT)) c.Select = 1;
        if (bit(R2)) c.R2 = 1;
        if (bit(L2)) c.L2 = 1;
        if (bit(A)) c.A = 1;
        if (bit(B)) c.B = 1;
        if (bit(X)) c.X = 1;
        if (bit(Y)) c.Y = 1;
        if (bit(DUP)) c.up = 1;
        if (bit(DRIGHT)) c.right = 1;
        if (bit(DDOWN)) c.down = 1;
        if (bit(DLEFT)) c.left = 1;

        const bool walk = !cfg_.deadman || bit(L1);
        key.RF_RX.lx = walk ? s.axes[0] : 0.0f;
        key.RF_RX.ly = walk ? s.axes[1] : 0.0f;
        key.RF_RX.rx = walk ? s.axes[2] : 0.0f;
        key.RF_RX.ry = walk ? s.axes[3] : 0.0f;
    }

    // for tests and the status line
    uint64_t accepted() const { return accepted_.load(); }
    uint64_t rejected() const { return rejected_.load(); }

private:
    struct Snapshot
    {
        bool have = false;
        bool connected = false;
        uint32_t buttons = 0;
        float axes[6] = {0, 0, 0, 0, 0, 0};
        std::chrono::steady_clock::time_point rx{};
    };

    static uint32_t rd32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
    static int16_t rd16(const uint8_t* p) { int16_t v; std::memcpy(&v, p, 2); return v; }

    void loop()
    {
        while (run_)
        {
            pollfd p{fd_, POLLIN, 0};
            if (::poll(&p, 1, 50) > 0 && (p.revents & POLLIN))
            {
                uint8_t b[128];
                sockaddr_in from{};
                socklen_t fl = sizeof from;
                ssize_t n = ::recvfrom(fd_, b, sizeof b, 0, reinterpret_cast<sockaddr*>(&from), &fl);
                if (n == static_cast<ssize_t>(kLen) && accept(b, from)) accepted_++;
                else rejected_++;
            }
        }
    }

    bool accept(const uint8_t* b, const sockaddr_in& from)
    {
        if (std::memcmp(b, "GPDK", 4) != 0 || b[4] != 1) return false;
        // constant-time key compare
        uint8_t diff = 0;
        for (int i = 0; i < 8; i++) diff |= static_cast<uint8_t>(b[26 + i] ^ cfg_.key[i]);
        if (diff != 0) return false;
        if (cfg_.allow_ip && from.sin_addr.s_addr != cfg_.allow_ip) return false;

        const uint32_t seq = rd32(b + 6);
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(mutex_);
        const bool live = snap_.have && std::chrono::duration<double>(now - snap_.rx).count() < 1.0;
        if (live)
        {
            if (from.sin_addr.s_addr != peer_ip_) return false;                     // another host mid-session
            if (static_cast<int32_t>(seq - last_seq_) <= 0) return false;           // stale or replayed
        }
        peer_ip_ = from.sin_addr.s_addr;
        last_seq_ = seq;
        snap_.have = true;
        snap_.connected = (b[5] & 1) != 0;
        snap_.buttons = rd32(b + 10);
        for (int i = 0; i < 6; i++)
        {
            float v = static_cast<float>(rd16(b + 14 + 2 * i)) / 32767.0f;
            snap_.axes[i] = v > 1.0f ? 1.0f : (v < -1.0f ? -1.0f : v);
        }
        snap_.rx = now;
        return true;
    }

    Config cfg_;
    int fd_ = -1;
    std::atomic<bool> run_{false};
    std::thread thread_;
    std::mutex mutex_;
    Snapshot snap_;
    in_addr_t peer_ip_ = 0;
    uint32_t last_seq_ = 0;
    bool was_fresh_ = false;                 // FSM thread only
    std::atomic<uint64_t> accepted_{0}, rejected_{0};
};

inline std::shared_ptr<PadInput> g_pad;     // null = pad off; set in main() before the FSM starts
