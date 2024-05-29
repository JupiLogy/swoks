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

class swoks():
    """
        Required inputs: prediction (value function), action, reward

        Output: environment label
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
        self.num_tasks = self.config["num_tasks"]
        self.last_task_change = 0
        self.adj = self.config["adj"]
        self.hist = {task: None for task in range(self.num_tasks)}
        self.visited_tasks = [0]
        self.L_D = self.config["h_len"]
        self.ts = 0
        self.pval = [0 for task in range(self.num_tasks)]
        self.sig = [0 for task in range(self.num_tasks)]
        self.emd_val = {task: [0] for task in range(self.num_tasks)}
        self.old_emd_val = {task: [0] for task in range(self.num_tasks)}
        self.old_hist = {task: None for task in range(self.num_tasks)}
        self.L_W = self.config["emd_limit"]
        self.old_raw_state = None
        self.current_task = 0
        self.tested_tasks = []
        self.total_len = self.L_D * (self.L_W+1)
        self.task_changing = False
        self.new_agent = False
        self.adopt_masks = adopt
        with open(moreconf.log_dir+"/json.json","w") as f:
            json.dump(self.config,f)

    def step(self, vals, r, a, supp=None, raw_state=None):
        """
            vals should be new agent internal state (value function) after
            receiving reward r, as a list with an element for each task.
            a is action taken.
            supp should be supplementary state info - for example, 2nd last layer of nn; or latent representation
        """
        self.ts += 1

        if type(a) != np.ndarray:
            a = [a]
        if self.hist[self.current_task] is None:
            # Initialise hists with first bits of data, ensuring correct array shapes, normalising wrt length of supp
            self.hist[self.current_task] = np.expand_dims(np.concatenate((a,[math.sqrt(len(supp))*r],supp)), axis=0)
        else:
            if len(self.hist[self.current_task]) > self.total_len and self.tested_tasks == []:
                # Cycle the arrays, to keep them finite length
                self.hist[self.current_task] = self.hist[self.current_task][-self.total_len:]
            # Append recent data to list/arrays
            self.hist[self.current_task] = np.append(self.hist[self.current_task], [np.concatenate((a,[math.sqrt(len(supp))*r],supp))], axis=0)

        if len(self.hist[self.current_task]) > self.L_D*3:
            for task in range(self.num_tasks):
                self.get_emd(task)

    def set_current_task(self, new_task):
        if (not new_task == self.current_task):
            self.last_task_change = self.ts
            self.current_task = new_task

    def get_emd(self, task):
        if self.ts % 240 != 0 or self.hist[task] is None or self.hist[self.current_task] is None:
            return
        elif task == self.current_task\
            and not (len(self.hist[task]) > self.L_D + 2): #and self.ts % self.emd_update_freq == 0):
            return
        if (task == self.current_task and self.tested_tasks == []) or self.old_hist[task] is None:
            task_hist = self.hist[task][:self.L_D]
        else:
            task_hist = self.old_hist[task]
        curr_task_hist = self.hist[self.current_task][-self.L_D:]
        weights1, weights2 = [np.ones(len(task_hist))/len(task_hist), np.ones(len(curr_task_hist))/len(curr_task_hist)]
        try:
            self.emd_val[task].append(ot.sliced_wasserstein_distance(
                    task_hist, curr_task_hist, a=weights1, b=weights2,
                    seed=self.seed))
            self.emd_val[task] = self.emd_val[task][-2*self.L_W:]
            if task != self.current_task:
                self.kol_smi(task,adjustment=self.adj)
            else:
                self.kol_smi(task,adjustment=self.adj)
        except RuntimeError:
            warnings.warn("EMD did not converge; if this happens often, increase EMD max iterations.\nCalculation skipped this timestep.", category=RuntimeWarning)
        except np.AxisError:
            print("axis error")
            import pdb; pdb.set_trace()
        except ValueError:
            print("value error")
            import pdb; pdb.set_trace()
        except TypeError:
            print("type error on EMD step. Current task:")
            print(self.current_task)
            import pdb; pdb.set_trace()

    def get_sig(self):
        sig = [(self.L_D**2)/(2*self.L_D) * self.emd_val[task] / self.L_D**1.5 for task in range(self.num_tasks)]
        return sig

    def kol_smi(self, task, adjustment=2):
        try:
            if self.old_emd_val[task] == [0]:
                if len(self.emd_val[task])>42:
                    old = self.emd_val[task][:-42]
                    new = self.emd_val[task][-42:]
                    self.pval[task] = stats.ks_2samp(adjustment*np.array(old),new,"greater").pvalue
            else:
                self.pval[task] = stats.ks_2samp(adjustment*np.array(self.old_emd_val[task]), self.emd_val[task][-42:],"greater").pvalue
        except:
            import pdb; pdb.set_trace()
        self.gen_task_label()
        if not self.tested_tasks == []:
            self.temp_change()

    def gen_task_label(self):
        if self.ts - self.last_task_change > self.stablephase and self.pval[self.current_task] < self.alpha:
            self.store_hist(self.current_task)
            self.tested_tasks += [self.current_task]
            print("storing history")

    def store_hist(self, task):
        self.old_emd_val[task] = self.emd_val[task][:-50]
        print("bounds")
        print(self.ts-len(self.hist[task]))
        print(self.ts-len(self.hist[task])+self.L_D)
        self.old_hist[task] = np.copy(self.hist[task][:self.L_D])
        print("current timestep")
        print(self.ts)
        self.task_changing = True

    def temp_change(self):
        if self.pval[self.current_task] > self.alpha * 1.5:
            print(f"assigning agent {self.current_task}")
            # If current task is right, we stop testing tasks (TODO: this isn't statistically powerful)
            self.tested_tasks = []
            self.hist[self.current_task] = np.concatenate((self.old_hist[self.current_task],\
                                                           self.hist[self.current_task][-self.L_D:]))
            return
        if self.ts - self.last_task_change > self.L_D*self.L_W:
            # if current task is not right, try the next untested task
            for task in self.visited_tasks:
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
            print("Creating new agent")

    def save(self, filename):
        file = open(filename, "wb")
        pickle.dump({"hist": self.hist, "old_hist": self.old_hist}, open(filename, "wb"))
        file.close()


"""
Copyright (C) 2024 Jeffery Dick

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
