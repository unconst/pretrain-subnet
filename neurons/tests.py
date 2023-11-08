import unittest
from collections import defaultdict

def get_wins(log, model_timestamps, valid_uids, random_pages):
  wins_per_page = defaultdict(int)

  for page in random_pages:
    losses = {uid: log[str(uid)][page]['losses'] for uid in valid_uids}

    sorted_losses = sorted(losses.items(), key=lambda x: (x[1], model_timestamps[x[0]]))

    for i, (uid, _) in enumerate(sorted_losses):
      if i == 0:
        wins_per_page[page] += 1
        
  for uid in valid_uids:
    log[str(uid)][page]['wins'] = [wins_per_page[page]]


class TestGetWins(unittest.TestCase):

  def test_single_page(self):
    log = {'1': {'page1': {'losses': 5}}, 
           '2': {'page1': {'losses': 10}}}
    model_timestamps = {'1': 100, '2': 200}
    uids = ['1', '2']
    pages = ['page1']
    get_wins(log, model_timestamps, uids, pages)
    self.assertEqual(log['1']['page1']['wins'], [1])
    self.assertEqual(log['2']['page1']['wins'], [0])


if __name__ == '__main__':
  unittest.main()