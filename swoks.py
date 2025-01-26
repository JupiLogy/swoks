"""
 ____    __      __  _____   __  __   ____
/\  _`\ /\ \  __/\ \/\  __`\/\ \/\ \ /\  _`\
\ \,\L\_\ \ \/\ \ \ \ \ \/\ \ \ \/'/'\ \,\L\_\
 \/_\__ \\ \ \ \ \ \ \ \ \ \ \ \ , <  \/_\__ \
   /\ \L\ \ \ \_/ \_\ \ \ \_\ \ \ \\`\  /\ \L\ \
   \ `\____\ `\___x___/\ \_____\ \_\ \_\\ `\____\
    \/_____/'\/__//__/  \/_____/\/_/\/_/ \/_____/
"""

import json
import warnings
import pickle
import numpy as np
import math
import ot
from scipy import stats
import context_detect, optimal_transport

class swoks():
    """
        Required inputs: observation INFO (latent representation),
                         action,
                         reward
    """
    def __init__(self, configs=None, adopt=False, moreconf=None):
        if configs:
            self.config = json.load(open(configs, 'r'))
        else:
            self.config = json.load(open('./configs/default.json', "r"))
        if moreconf is not None:
            self.seed=moreconf.seed
        else:
            self.seed=1
        np.random.seed(self.seed)
        if self.config["alpha"] is not None:
            self.alpha = self.config["alpha"]
        else:
            self.alpha = 0.001
        if self.config["stablephase"] is not None:
            self.stablephase = self.config["stablephase"]
        else:
            self.stablephase = 36000
        self.task_list = [0]
        self.last_task_change = 0
        self.visited_tasks = [0]
        self.L_D = self.config["h_len"]
        self.ts = 0
        self.pval = [0 for task in range(self.num_tasks)]
        self.L_W = self.config["emd_limit"]
        self.old_raw_state = None
        self.current_task = 0
        self.tested_tasks = []
        self.task_changing = False
        self.new_agent = False
        self.adopt_masks = adopt
        self.hist = context_detect.HistoryManager(h_len=self.L_D, num_wins=self.L_W)
        self.wass_hist = context_detect.HistoryManager(h_len=self.L_W, num_wins=1)
        if self.config["context_detector"] in ["CUSUM","FELT"]:
            needs_updating = True
        else:
            needs_updating = False

        ot_dict = {"SlicedWass":optimal_transport.sliced_wass(self.seed),}
        cd_dict = {"KS":context_detect.cd_ks,
                   "CvM": context_detect.cd_cvm,
                   "AD": context_detect.cd_ad,}
        try:
            self.ot_alg=ot_dict[self.config["ot_alg"]]
        except KeyError:
            raise KeyError("check ot_alg in config matches \"SlicedWass\".")

        self.context_detector = cd_dict[self.config["context_detector"]](
            self.hist,
            wass_hist=self.wass_hist,
            ot_alg=self.config["ot_alg"],
            #needs_updating=needs_updating,
            adj=self.config["adj"]
        )
        with open(moreconf.log_dir+"/json.json","w") as f:
            json.dump(self.config,f)

    def step(self, r, a, supp=None, raw_state=None):
        """
            reward is r.
            a is action taken.
            supp should be supplementary state info - for example,
                2nd last layer of nn; or latent representation
        """
        if supp is None:
            raise AssertionError("swoks needs supplementary state info"+\
                                 "from your neural network!")

        self.ts += 1

        # Update history manager
        if type(a) != np.ndarray:
            a = [a]
        self.hist.update([np.concatenate((a,[math.sqrt(len(supp))*r],supp))])

        # recalculate p-value.
        if self.ts % self.L_D == 0:
            for task in self.task_list:
                self.context_detector.pval(task=task)
            if not self.tested_tasks == []:
                self.temp_change()

    def set_current_task(self, new_task):
        if (not new_task == self.current_task):
            self.last_task_change = self.ts
            self.current_task = new_task

    def gen_task_label(self):
        if self.ts - self.last_task_change > self.stablephase and\
           self.pval[self.current_task] < self.alpha:
            # New task detected.
            self.tested_tasks += [self.current_task]

    def store_hist(self):
        self.task_changing = True

    def temp_change(self):
        self.context_detector.testing = True
        if self.pval[self.current_task] > self.alpha * 1.5:
            print(f"assigning agent {self.current_task}")
            # If current task is right, we stop testing tasks
            # (TODO: this isn't statistically powerful)
            self.context_detector.testing = False
            self.context_detector.set_task(self.current_task)
            self.tested_tasks = []
            self.hist[self.current_task] =\
                np.concatenate((self.old_hist[self.current_task],\
                                self.hist[self.current_task][-self.L_D:]))
            return
        if self.ts - self.last_task_change > self.L_D*self.L_W:
            # if current task is not right, try the next untested task
            for task in self.task_list:
                if task not in self.tested_tasks:
                    print(f"testing agent {task}")
                    self.set_current_task(task)
                    self.tested_tasks += [task]
                    return
            # Only get here if no existing task pinged
            self.set_current_task(task+1)
            self.visited_tasks += [task+1]
            self.tested_tasks = []
            self.new_agent = True
            self.context_detector.testing = False
            self.context_detector.set_task(task+1)
            print("Creating new agent")

    def save(self, filename):
        file = open(filename, "wb")
        pickle.dump({"hist": self.hist, "old_hist": self.old_hist},\
                    open(filename, "wb"))
        file.close()


"""
Copyright (C) 2024-2025 Jeffery Dick

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
