from aha.memory.store import Episode, EpisodicMemory, SemanticMemory


def test_memory_retrieval_and_compression():
    episode = Episode({"x": 1}, {"x": 2}, {"x": 2}, 0)
    episodic = EpisodicMemory(capacity=2)
    semantic = SemanticMemory()

    episodic.add(episode)
    semantic.compress_episode(episode)

    assert episodic.retrieve_recent(1)[0] == episode
    assert semantic.retrieve_common(1)[0][1] == 1
