import os

from kivy.app import App
from kivy.properties import ObjectProperty, StringProperty, BoundedNumericProperty, ListProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.core.window import Window

import utils.utils
from model.model import Model
from utils.utils import load_persons, load_availabilities, generate_availability_matrix, save_schedule, load_shifts


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
    time_solver = StringProperty('')

    persons = None
    shifts = None
    ref_time = None
    availabilities = None
    availability_matrix = None

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
        print(s.is_major_only for s in self.shifts)
        print([s.nb_persons for s in self.shifts])
        print([[s.start, s.end] for s in self.shifts])

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

    def compute_and_save(self):
        if self.persons_file == '':
            self.show_file_not_selected('personnes')
        elif self.shifts_file == '':
            self.show_file_not_selected('shifts')
        elif self.availabilities_file == '':
            self.show_file_not_selected('disponibilités')
        elif self.save_file == '':
            self.show_file_not_selected('résultats')
        elif int(self.min_nb_shift.text) > int(self.max_nb_shift.text) or int(self.min_nb_shift.text) < 0 or int(self.max_nb_shift.text) < 0:
            self.show_nb_shifts_error()
        else:
            # Compute the availability matrix
            self.availability_matrix = generate_availability_matrix(self.persons,
                                                                    self.shifts,
                                                                    self.availabilities,
                                                                    self.ref_time)

            # Instantiate the model
            model = Model(min_nb_shifts=int(self.min_nb_shift.text), max_nb_shifts=int(self.max_nb_shift.text))
            model.build_model(self.persons, self.shifts, self.availability_matrix)

            # Kick the computations
            assignments, status_str, wall_time = model.solve()

            if status_str == model.status_str_invalid:
                self.show_invalid_model_error()
            elif status_str == model.status_str_infeasible:
                self.show_infeasible_solution_error()
            else:
                # Save the results
                save_schedule(self.save_file, assignments)

                # Update time solver
                self.time_solver = f'{wall_time}'

                self.show_saved_successfully()


class PlanningApp(App):
    def build(self):
        Window.size = (1000, 750)
        Window.minimum_width, Window.minimum_height = Window.size
        self.icon = 'icon.jpeg'
        return Root()


if __name__ == '__main__':
    PlanningApp().run()