// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include <atomic>
#include <chrono>
#include <vector>



class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        // set gain
        const int n = env->robot->data.joint_stiffness.size();
        from_q.resize(n); from_kp.resize(n); from_kd.resize(n);
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            from_q[i] = lowcmd->msg_.motor_cmd()[i].q();
            from_kp[i] = lowcmd->msg_.motor_cmd()[i].kp();
            from_kd[i] = lowcmd->msg_.motor_cmd()[i].kd();
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
            // lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            // lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            // lowcmd->msg_.motor_cmd()[i].dq() = 0;
            // lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }
        // Reset the declared booleans.
        first_step_done = false;
        switch_started = false;
        idle_started = false;

        env->robot->update();
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();
                first_step_done = true;

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    // Declare the policy thread.
    std::thread policy_thread;
    bool policy_thread_running = false;
    // Declare q, kp, and kd variables.
    std::vector<float> from_q, from_kp, from_kd;
    // Declare the policy switch variables.
    std::chrono::steady_clock::time_point switch_start;
    bool switch_started = false;
    static constexpr double SWITCH_TIME = 0.3;
    std::atomic<bool> first_step_done{false};
    // Declare velocity idle timer.
    std::chrono::steady_clock::time_point idle_start;
    bool idle_started = false;
    // Declare the idle time.
    static constexpr double IDLE_TIME = 0.5;
};

REGISTER_FSM(State_RLBase)
