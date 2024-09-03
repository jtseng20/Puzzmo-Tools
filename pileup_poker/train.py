from ppo.ppo import PPO

if __name__ == "__main__":
    gen_size = 5000
    gen_steps = 60
    buffer_size = gen_size * gen_steps * 1.5 # how many past generations to hold in memory before cycling out
    ppo = PPO(gen_batch_size=gen_size, max_gen_steps=gen_steps, 
                  mini_batch_size=4096, num_epochs=1, 
                  dirichlet_eps=0.85, buffer_len=buffer_size, 
                  device="cuda", wandb_log=True)
    ppo.train_loop(num_iters=10000)