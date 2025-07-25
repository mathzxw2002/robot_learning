# TODO: 
1, add slam and nav2, to enable the moving function of lekiwi \
2, add a third camera \
3, using other policies \
4, multiple gpu training \
5, 


# robot_learning
robot_learning


python3 -m lerobot.scripts.eval --policy.path=lerobot/diffusion_pusht --env.type=pusht --eval.batch_size=10 --eval.n_episodes=10 --policy.use_amp=false --policy.device=cpu
