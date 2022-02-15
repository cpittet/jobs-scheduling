import asyncio
import os
import threading

from kivy.app import App
from kivy.clock import mainthread, Clock
from kivy.core.window import Window
from kivy.event import EventDispatcher
from kivy.properties import ObjectProperty, StringProperty, ListProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from ortools.sat.python import cp_model

import utils.utils
from model.model import create_and_solve, SolutionSaverCallback
from utils.utils import load_persons, load_availabilities, generate_availability_matrix, load_shifts, save_schedule


IMPOSSIBLE = 1


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    file_filter = ListProperty(['*.csv'])


class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)


class MessageDialog(FloatLayout):
    ok = ObjectProperty(None)
    text = StringProperty('')


class LoopWorker(EventDispatcher):
    # Event for this dispatcher
    __events__ = ('on_new_solution',)

    def __init__(self):
        super().__init__()

        self._thread = threading.Thread(target=self.run_loop)  # note the Thread target here
        self._thread.daemon = True
        self.loop = None
        self.task = None
        self.futur = None

    def start(self, min_nb_shift, max_nb_shift, persons, shifts, availability_matrix, save_file, max_solver_time):
        self.min_nb_shift = min_nb_shift
        self.max_nb_shift = max_nb_shift
        self.persons = persons
        self.shifts = shifts
        self.availability_matrix = availability_matrix
        self.save_file = save_file
        self.max_solver_time = max_solver_time

        self._thread.start()

    def run_loop(self):
        self.loop = asyncio.get_event_loop_policy().new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Kick the actual computations
        if self.futur is not None:
            self.futur.cancel()
        self.futur = self.loop.create_future()

        self.task = self.loop.create_task(self.compute(self.futur))
        self.loop.run_until_complete(self.futur)

    async def compute(self, futur):
        self.callback = SolutionSaverCallback(self)
        status = create_and_solve(self.min_nb_shift,
                                  self.max_nb_shift,
                                  self.persons,
                                  self.shifts,
                                  self.availability_matrix,
                                  self.callback,
                                  self.max_solver_time)

        futur.set_result(status)
        save_schedule(self.save_file, self.callback.best_assignments)

    @mainthread
    def post_count(self, nb):
        self.dispatch('on_new_solution', nb)

    def on_new_solution(self, *_):
        pass

    def stop(self):
        self.task.cancel()
        self._thread.join()

        if self.futur.done() and self.callback.best_assignments is not None:
            save_schedule(self.save_file, self.callback.best_assignments)
        return self.futur.result()


class Root(FloatLayout):
    load_person_file = ObjectProperty(None)
    load_shift_file = ObjectProperty(None)
    save_schedule_file = ObjectProperty(None)
    min_nb_shift = ObjectProperty(None)
    max_nb_shift = ObjectProperty(None)
    persons_file = StringProperty('')
    shifts_file = StringProperty('')
    availabilities_file = StringProperty('')
    save_file = StringProperty('')
    nb_solutions = StringProperty('0')
    max_solver_time = ObjectProperty(None)
    sol_status = StringProperty('')

    persons = None
    shifts = None
    ref_time = None
    availabilities = None
    availability_matrix = None
    computations_task = None
    loop_worker = None

    def close_popup(self):
        self.popup.dismiss()

    def show_load_persons(self):
        content = LoadDialog(load=self.load_persons, cancel=self.close_popup)
        self.popup = Popup(title='Sélectionner les personnes',
                           content=content,
                           size_hint=(0.8, 0.8),
                           auto_dismiss=False)

        self.popup.open()

    def show_load_shifts(self):
        content = LoadDialog(load=self.load_shifts, cancel=self.close_popup)
        self.popup = Popup(title='Sélectionner les shifts',
                           content=content,
                           size_hint=(0.8, 0.8),
                           auto_dismiss=False)

        self.popup.open()

    def show_load_availabilities(self):
        content = LoadDialog(load=self.load_availabilities, cancel=self.close_popup)
        self.popup = Popup(title='Sélectionner les disponibilités',
                           content=content,
                           size_hint=(0.8, 0.8),
                           auto_dismiss=False)

        self.popup.open()

    def show_save(self):
        content = SaveDialog(save=self.select_save, cancel=self.close_popup)
        self.popup = Popup(title='Sélectionner le fichier de destination',
                           content=content,
                           size_hint=(0.8, 0.8),
                           auto_dismiss=False)

        self.popup.open()

    def load_persons(self, path, filename):
        self.persons_file = os.path.join(path, filename[0])
        self.persons = load_persons(self.persons_file)

        self.close_popup()

        error_msg = None
        if self.persons == utils.utils.INVALID_COLUMNS:
            error_msg = f'Le fichier {self.persons_file} ne contient pas les bonnes colonnes ' \
                        f'("nom", "age", "responsable")! Veuillez contrôler son contenu.'
        elif self.persons == utils.utils.NOT_UNIQUE_DATA:
            error_msg = f'Le fichier {self.persons_file} contient plusieurs fois la même personne !' \
                        f' Contrôlez son contenu.'
        elif self.persons == utils.utils.INVALID_MAJOR_DATA:
            error_msg = f'La colonne "responsable" du fichier {self.persons_file} contient des données différentes de' \
                        f'"oui" ou "non" ! Veuillez contrôler son contenu.'

        if error_msg is not None:
            self.persons = None
            self.persons_file = ''
            content = MessageDialog(text=error_msg, ok=self.close_popup)
            self.popup = Popup(title='Données invalides',
                               content=content,
                               size_hint=(0.8, 0.8))
            self.popup.open()

    def load_shifts(self, path, filename):
        self.shifts_file = os.path.join(path, filename[0])
        self.shifts, self.ref_time = load_shifts(self.shifts_file)

        self.close_popup()

        error_msg = None
        if self.shifts == utils.utils.INVALID_COLUMNS:
            error_msg = f'Le fichier {self.shifts_file} ne contient pas les bonnes colonnes ' \
                        f'("id", "debut", "fin", "nombre", "majeur") ! Veuillez contrôler son contenu.'
        elif self.shifts == utils.utils.NOT_UNIQUE_DATA:
            error_msg = f'Le fichier {self.shifts_file} contient plusieurs fois le même shift id' \
                        f' ! Contrôlez son contenu.'
        elif self.shifts == utils.utils.INVALID_MAJOR_DATA:
            error_msg = f'La colonne "majeur" du fichier {self.shifts_file} contient des données différentes de' \
                        f'"oui" ou "non" ! Veuillez contrôler son contenu.'
        elif self.shifts == utils.utils.INVALID_DATE_DATA:
            error_msg = f"Le début d'un shift doit être avant sa fin ! (Contrôler en particulier les shifts " \
                        f"commençant/finissant à minuit)"

        if error_msg is not None:
            self.shifts = None
            self.shifts_file = ''
            content = MessageDialog(text=error_msg, ok=self.close_popup)
            self.popup = Popup(title='Données invalides',
                               content=content,
                               size_hint=(0.8, 0.8))
            self.popup.open()

    def load_availabilities(self, path, filename):
        self.availabilities_file = os.path.join(path, filename[0])
        self.availabilities = load_availabilities(self.availabilities_file)

        self.close_popup()

        error_msg = None
        if self.availabilities == utils.utils.INVALID_COLUMNS:
            error_msg = f'Le fichier {self.availabilities_file} ne contient pas les bonnes colonnes ' \
                        f'("nom", "debut", "fin") ! Veuillez contrôler son contenu.'
        elif self.availabilities == utils.utils.INVALID_DATE_DATA:
            error_msg = f"Le début d'une période doit être avant sa fin ! (Contrôler en particulier les périodes " \
                        f"commençant/finissant à minuit)"

        if error_msg is not None:
            self.availabilities = None
            self.availabilities_file = ''
            content = MessageDialog(text=error_msg, ok=self.close_popup)
            self.popup = Popup(title='Données invalides',
                               content=content,
                               size_hint=(0.8, 0.8))
            self.popup.open()

    def select_save(self, path, filename):
        if not filename.endswith('.csv'):
            filename = f'{filename}.csv'

        self.save_file = os.path.join(path, filename)

        self.close_popup()

    def show_file_not_selected(self, text):
        message = f"Il faut d'abord sélectionner le fichiers contenant les {text} !"
        content = MessageDialog(text=message, ok=self.close_popup)
        self.popup = Popup(title='Fichier manquant',
                           content=content,
                           size_hint=(0.6, 0.6))

        self.popup.open()

    def show_nb_shifts_error(self):
        message = "Le nombre minimum de shifts par personne doit être plus petit que le maximum et être positif !"
        content = MessageDialog(text=message, ok=self.close_popup)
        self.popup = Popup(title='Erreur : nombre de shifts',
                           content=content,
                           size_hint=(0.6, 0.6))

        self.popup.open()

    def show_invalid_model_error(self):
        message = "Le modèle est invalide. Contacter le développeur."
        content = MessageDialog(text=message, ok=self.close_popup)
        self.popup = Popup(title='Erreur : modèle invalide',
                           content=content,
                           size_hint=(0.6, 0.6))

        self.popup.open()

    def show_infeasible_solution_error(self):
        message = "Impossible de trouver une solution. " \
                  "Essayez de réduire le nombre minimum de shifts par personne (par exemple 0 ou 1) " \
                  "et augmenter le nombre maximum de shifts par personne et relancer les calculs."
        content = MessageDialog(text=message, ok=self.close_popup)
        self.popup = Popup(title='Pas de solution',
                           content=content,
                           size_hint=(0.6, 0.6))

        self.popup.open()

    def show_saved_successfully(self):
        message = f"Calculs terminés avec succès ! Résultats sauvegardés dans {self.save_file}"
        content = MessageDialog(text=message, ok=self.close_popup)
        self.popup = Popup(title='Solution trouvée',
                           content=content,
                           size_hint=(0.6, 0.6))

        self.popup.open()

    def show_no_max_solver_time(self):
        message = f"Vous devez spécifier un temps maximal de calculs (en minutes) strictement positif !"
        content = MessageDialog(text=message, ok=self.close_popup)
        self.popup = Popup(title='Information manquante',
                           content=content,
                           size_hint=(0.6, 0.6))

        self.popup.open()

    def show_no_sol_yet(self):
        message = f"Aucune solution trouvée pour le moment. Essayez d'augmenter le temps de calcul !"
        content = MessageDialog(text=message, ok=self.close_popup)
        self.popup = Popup(title='Pas encore de solution trouvée',
                           content=content,
                           size_hint=(0.6, 0.6))

        self.popup.open()

    def compute_and_save(self):
        if self.persons_file == '':
            self.show_file_not_selected('personnes')
        elif self.shifts_file == '':
            self.show_file_not_selected('shifts')
        elif self.availabilities_file == '':
            self.show_file_not_selected('disponibilités')
        elif self.save_file == '':
            self.show_file_not_selected('résultats')
        elif self.min_nb_shift.text == '' or self.max_nb_shift.text == '' or \
                int(self.min_nb_shift.text) > int(self.max_nb_shift.text) or \
                int(self.min_nb_shift.text) < 0 or int(self.max_nb_shift.text) < 0:
            self.show_nb_shifts_error()
        elif self.max_solver_time.text == '' or int(self.max_solver_time.text) <= 0:
            self.show_no_max_solver_time()
        else:
            # Compute the availability matrix
            self.availability_matrix = generate_availability_matrix(self.persons,
                                                                    self.shifts,
                                                                    self.availabilities,
                                                                    self.ref_time)

            self.sol_status = ''

            # Kick the computations
            if self.loop_worker is None:
                self.loop_worker = LoopWorker()

                def display_solutions_count(instance, nb):
                    self.nb_solutions = str(nb)

                self.loop_worker.bind(on_new_solution=display_solutions_count)
                self.loop_worker.start(int(self.min_nb_shift.text),
                                       int(self.max_nb_shift.text),
                                       self.persons,
                                       self.shifts,
                                       self.availability_matrix,
                                       self.save_file,
                                       int(self.max_solver_time.text))

    def stop_computations(self):
        if self.loop_worker is not None:
            result = self.loop_worker.stop()
            self.loop_worker = None

            if result == cp_model.INFEASIBLE:
                self.show_infeasible_solution_error()
            elif int(self.nb_solutions) > 0:
                self.show_saved_successfully()
            else:
                self.show_no_sol_yet()

    def check_for_status(self, dt):
        if self.loop_worker is not None:
            status_futur = self.loop_worker.futur

            if status_futur.done():
                status = status_futur.result()

                if status == cp_model.OPTIMAL:
                    status_str = 'Optimal'
                elif status == cp_model.FEASIBLE:
                    status_str = 'Faisable'
                elif status == cp_model.INFEASIBLE:
                    status_str = 'Impossible'
                print('setting status')
                self.sol_status = status_str

                self.stop_computations()


class PlanningApp(App):
    def build(self):
        Window.size = (1000, 750)
        Window.minimum_width, Window.minimum_height = Window.size
        self.icon = 'icon.jpeg'
        root = Root()
        Clock.schedule_interval(root.check_for_status, 1.0)
        return root


if __name__ == '__main__':
    PlanningApp().run()
