// test_pad_input.cpp - host test for pad_input.h (no robot, no DDS). It uses the SDK's real UnitreeJoystick.
//   g++ -std=c++17 -I deploy/include -I <unitree_sdk2>/include test_pad_input.cpp -o test_pad_input -lspdlog -lfmt -lpthread
#include <arpa/inet.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "pad_input.h"

using unitree::common::REMOTE_DATA_RX;
using unitree::common::UnitreeJoystick;

static int g_fail = 0, g_pass = 0;
#define CHECK(cond, msg) do { if (cond) { g_pass++; printf("  ok    %s\n", msg); } else { g_fail++; printf("  FAIL  %s\n", msg); } } while (0)

static const uint8_t KEY[8] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef};
static uint32_t g_seq = 100;

struct Pad  // a sender: builds and sends GPDK packets to 127.0.0.1:port
{
    int fd, port;
    explicit Pad(int p) : fd(::socket(AF_INET, SOCK_DGRAM, 0)), port(p) {}
    ~Pad() { ::close(fd); }
    void raw(const uint8_t* b, size_t n)
    {
        sockaddr_in to{};
        to.sin_family = AF_INET;
        to.sin_port = htons(static_cast<uint16_t>(port));
        inet_pton(AF_INET, "127.0.0.1", &to.sin_addr);
        ::sendto(fd, b, n, 0, reinterpret_cast<sockaddr*>(&to), sizeof to);
    }
    void send(uint32_t buttons, float lx = 0, float ly = 0, float rx = 0, float ry = 0, bool connected = true,
              const uint8_t* key = KEY, int32_t seq = -1, const char* magic = "GPDK")
    {
        uint8_t b[34] = {0};
        std::memcpy(b, magic, 4);
        b[4] = 1;
        b[5] = connected ? 1 : 0;
        uint32_t s = seq < 0 ? ++g_seq : static_cast<uint32_t>(seq);
        std::memcpy(b + 6, &s, 4);
        std::memcpy(b + 10, &buttons, 4);
        int16_t ax[6] = {static_cast<int16_t>(lx * 32767), static_cast<int16_t>(ly * 32767), static_cast<int16_t>(rx * 32767),
                         static_cast<int16_t>(ry * 32767), 0, 0};
        std::memcpy(b + 14, ax, 12);
        std::memcpy(b + 26, key, 8);
        raw(b, sizeof b);
    }
};

static uint32_t B(std::initializer_list<int> bits) { uint32_t v = 0; for (int b : bits) v |= 1u << b; return v; }

// one 1 kHz-style FSM tick: remote bytes -> merge -> extract (exactly what FSMState::pre_run does)
static void tick(PadInput& pad, UnitreeJoystick& joy, REMOTE_DATA_RX remote = REMOTE_DATA_RX{})
{
    pad.merge(remote);
    joy.extract(remote);
}
static void run_ms(PadInput& pad, UnitreeJoystick& joy, int ms, const std::function<void(int)>& each = nullptr,
                   const REMOTE_DATA_RX& remote = REMOTE_DATA_RX{})
{
    for (int i = 0; i < ms; i++)
    {
        if (each) each(i);
        tick(pad, joy, remote);
        usleep(1000);
    }
}
static REMOTE_DATA_RX zero() { REMOTE_DATA_RX r; std::memset(&r, 0, sizeof r); return r; }

int main()
{
    spdlog::set_level(spdlog::level::warn);
    PadInput::Config c;
    c.port = 19872;
    std::memcpy(c.key, KEY, 8);
    c.timeout_s = 0.25;
    PadInput pad(c);
    Pad tx(c.port);
    UnitreeJoystick joy;
    REMOTE_DATA_RX r0 = zero();

    printf("== nothing received yet: the remote passes through untouched\n");
    REMOTE_DATA_RX rem = zero(); rem.RF_RX.btn.components.A = 1; rem.RF_RX.lx = 0.5f;
    run_ms(pad, joy, 50, nullptr, rem);
    CHECK(joy.A.pressed, "remote A still works");
    CHECK(joy.lx() > 0.1f, "remote stick still works (pad not fresh)");

    printf("== a held pad button gives ONE on_pressed, not one per tick (pad sends at 100 Hz)\n");
    {
        UnitreeJoystick j2; int on_press = 0, on_rel = 0;
        run_ms(pad, j2, 400, [&](int i) { if (i % 10 == 0) tx.send(B({PadInput::A})); });
        // count edges over another 300 ms of the same held button
        for (int i = 0; i < 300; i++) { if (i % 10 == 0) tx.send(B({PadInput::A})); tick(pad, j2); on_press += j2.A.on_pressed; on_rel += j2.A.on_released; usleep(1000); }
        CHECK(j2.A.pressed, "A pressed from the pad");
        CHECK(on_press == 0 && on_rel == 0, "no extra on_pressed / on_released while it is held (no flip-flop)");
        UnitreeJoystick j3; int edges = 0;
        run_ms(pad, j3, 350);                                 // idle > 250 ms: the pad goes silent
        for (int i = 0; i < 300; i++) { if (i % 10 == 0) tx.send(B({PadInput::A})); tick(pad, j3); edges += j3.A.on_pressed; usleep(1000); }
        CHECK(edges == 1, "exactly one on_pressed when the button goes down");
    }

    printf("== hold time (the DSL's LT(2s) style) keeps counting through the pad\n");
    {
        UnitreeJoystick j; run_ms(pad, j, 350);                 // pad silent first
        auto t0 = std::chrono::steady_clock::now();
        run_ms(pad, j, 700, [&](int i) { if (i % 10 == 0) tx.send(B({PadInput::A})); });
        float elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count();
        printf("        (held %.2f s of wall clock, pressed_time %.2f s)\n", elapsed, j.A.pressed_time);
        CHECK(j.A.pressed && j.A.pressed_time > elapsed - 0.12f && j.A.pressed_time < elapsed + 0.02f,
              "pressed_time grows continuously and matches the real hold time");
    }

    printf("== combos arrive like the remote's: L2 + B\n");
    {
        UnitreeJoystick j; run_ms(pad, j, 350);
        run_ms(pad, j, 60, [&](int i) { if (i % 10 == 0) tx.send(B({PadInput::L2, PadInput::B})); });
        CHECK(j.B.pressed && j.LT.pressed, "pad L2 + B reach the joystick (LT ramps up like with the remote)");
    }

    printf("== the Unitree remote's buttons are OR-ed (the spotter's L2 + B always works)\n");
    {
        UnitreeJoystick j; REMOTE_DATA_RX rm = zero(); rm.RF_RX.btn.components.L2 = 1; rm.RF_RX.btn.components.B = 1;
        run_ms(pad, j, 60, [&](int i) { if (i % 10 == 0) tx.send(B({PadInput::Y})); }, rm);
        CHECK(j.B.pressed && j.Y.pressed, "remote B and pad Y are both seen");
    }

    printf("== sticks: the pad owns them only while fresh, and only with L1 held (deadman)\n");
    {
        UnitreeJoystick j; run_ms(pad, j, 350);
        REMOTE_DATA_RX rm = zero(); rm.RF_RX.lx = 1.0f;
        run_ms(pad, j, 400, [&](int i) { if (i % 10 == 0) tx.send(0, 0, 1.0f); }, rm);
        CHECK(std::fabs(j.ly()) < 0.05f, "pad stick forward WITHOUT L1: ignored");
        CHECK(std::fabs(j.lx()) < 0.05f, "remote stick ignored while the pad is fresh");
        run_ms(pad, j, 400, [&](int i) { if (i % 10 == 0) tx.send(B({PadInput::L1}), 0, 1.0f); });
        CHECK(j.ly() > 0.9f, "pad stick forward WITH L1: forward (ly -> 1)");
        run_ms(pad, j, 400, [&](int i) { if (i % 10 == 0) tx.send(B({PadInput::L1}), 0, 0.0f); });
        CHECK(std::fabs(j.ly()) < 0.05f, "stick released: back to zero");
    }

    printf("== silence: after 250 ms the pad counts as gone, the remote is back\n");
    {
        UnitreeJoystick j; run_ms(pad, j, 350);
        run_ms(pad, j, 200, [&](int i) { if (i % 10 == 0) tx.send(B({PadInput::A, PadInput::L1}), 0, 1.0f); });
        CHECK(j.A.pressed && j.ly() > 0.5f, "pad active (A held, walking)");
        REMOTE_DATA_RX rm = zero(); rm.RF_RX.lx = 0.6f;
        run_ms(pad, j, 600, nullptr, rm);                     // the laptop script died
        CHECK(!j.A.pressed, "pad button released after the silence");
        CHECK(std::fabs(j.ly()) < 0.05f, "pad's forward command gone: velocity back to neutral");
        CHECK(j.lx() > 0.3f, "the remote's stick works again");
    }

    printf("== disconnected flag: no pad input\n");
    {
        UnitreeJoystick j; run_ms(pad, j, 350);
        run_ms(pad, j, 100, [&](int i) { if (i % 10 == 0) tx.send(B({PadInput::A}), 0, 0, 0, 0, false); });
        CHECK(!j.A.pressed, "connected=0 packets are ignored");
    }

    printf("== authentication: wrong key, magic, length, replay\n");
    {
        UnitreeJoystick j; run_ms(pad, j, 350);
        uint8_t bad[8]; std::memcpy(bad, KEY, 8); bad[7] ^= 1;
        uint64_t acc0 = pad.accepted();
        run_ms(pad, j, 100, [&](int i) { if (i % 10 == 0) tx.send(B({PadInput::A}), 0, 0, 0, 0, true, bad); });
        CHECK(!j.A.pressed && pad.accepted() == acc0, "wrong key: dropped");
        run_ms(pad, j, 100, [&](int i) { if (i % 10 == 0) tx.send(B({PadInput::A}), 0, 0, 0, 0, true, KEY, -1, "GPAD"); });
        CHECK(!j.A.pressed, "wrong magic (the unauthenticated pad-link format): dropped");
        uint8_t shortp[10] = {0}; tx.raw(shortp, sizeof shortp); usleep(30000);
        CHECK(!j.A.pressed, "short packet: dropped");
        run_ms(pad, j, 100, [&](int i) { if (i % 10 == 0) tx.send(B({PadInput::A})); });
        CHECK(j.A.pressed, "a valid packet works again");
        uint32_t cur = g_seq;
        run_ms(pad, j, 100, [&](int i) { if (i % 10 == 0) tx.send(B({PadInput::X}), 0, 0, 0, 0, true, KEY, static_cast<int32_t>(cur - 5)); });
        CHECK(!j.X.pressed, "replayed (old) sequence number inside a live session: dropped");
        run_ms(pad, j, 1300);                                  // > 1 s of silence: a restarted sender may begin again
        run_ms(pad, j, 100, [&](int i) { if (i % 10 == 0) tx.send(B({PadInput::X}), 0, 0, 0, 0, true, KEY, 5); });
        CHECK(j.X.pressed, "restarted sender (sequence 5) accepted after > 1 s of silence");
    }

    printf("== F1 / F2 are never set from the pad\n");
    {
        UnitreeJoystick j; run_ms(pad, j, 350);
        run_ms(pad, j, 100, [&](int i) { if (i % 10 == 0) tx.send(0xFFFFFFFFu); });
        CHECK(!j.F1.pressed && !j.F2.pressed, "F1 / F2 stay off even with every pad bit set");
        CHECK(j.A.pressed && j.up.pressed && j.RB.pressed, "while the real buttons come through");
    }

    // allow-ip: a second listener that only accepts 10.9.9.9 must ignore 127.0.0.1
    {
        PadInput::Config c2 = c; c2.port = 19873; inet_pton(AF_INET, "10.9.9.9", &c2.allow_ip);
        PadInput pad2(c2); Pad tx2(c2.port); UnitreeJoystick j;
        printf("== --pad-allow-ip\n");
        run_ms(pad2, j, 100, [&](int i) { if (i % 10 == 0) tx2.send(B({PadInput::A})); });
        CHECK(!j.A.pressed && pad2.accepted() == 0, "packets from another IP are dropped");
    }

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
