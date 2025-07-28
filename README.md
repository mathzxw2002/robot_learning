# TODO: 
- [ ] design the structure of the robot, decide the scenario 
- [ ] add slam and nav2, to enable the moving function of lekiwi 
- [ ] using other policies 
- [ ] multiple gpu training  
- [ ] mcp  


# robot_learning
robot_learning


python3 -m lerobot.scripts.eval --policy.path=lerobot/diffusion_pusht --env.type=pusht --eval.batch_size=10 --eval.n_episodes=10 --policy.use_amp=false --policy.device=cpu
