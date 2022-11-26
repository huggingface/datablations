### Annoying NCCL error without NCCL_SOCKET_IFNAME

```
work = default_pg.barrier(opts=opts)
RuntimeError: NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1191, unhandled system error, NCCL version 2.11.4
ncclSystemError: System call (e.g. socket, malloc) or external library call failed or device error. It can be also caused by unexpected exit of a remote peer, you can check NCCL warnings for failure reason and see if there is connection closure by a peer.
ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: 1) local_rank: 0 (pid: 15855) of binary: /pfs/lustrep4/users/nouatazi/projects/venv/bin/python
```

Happens if you remove `export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3`, cf [Nouamane](https://huggingface.slack.com/archives/C0425P7N73M/p1665494464503129)
