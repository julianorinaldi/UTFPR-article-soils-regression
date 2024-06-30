from dto.ConfigModelDTO import ConfigModelDTO
from dto.NormalizeEnum import convert_normalize_set_enum
from dto.ModelSetEnum import convert_model_set_enum
from shared.infrastructure.log import LoggingPy


def get_config_from_args(args, logger: LoggingPy = None):
    if not args.name:
        raise Exception("Há parâmetros faltantes. Utilize -h ou --help para ajuda!")

    # Definindo Modelo de TransferLearning e Configurações
    model_set_enum = convert_model_set_enum(args.model)
    image_dimension_x = 256
    image_dimension_y = 256
    qtd_canal_color = 3
    path_csv = ""
    dir_base_img = ""
    normalize_enum = convert_normalize_set_enum(args.normalize)

    config = ConfigModelDTO(logger=logger, model_set_enum=model_set_enum, path_csv=path_csv, dir_base_img=dir_base_img,
                            image_dimension_x=image_dimension_x, image_dimension_y=image_dimension_y,
                            channel_colors=qtd_canal_color, amount_images_train=args.amount_image_train,
                            amount_images_test=args.amount_image_test, log_level=args.log_level,
                            args_name_model=args.name, args_normalize=normalize_enum, args_trainable=args.trainable,
                            args_separed=args.separed, args_preprocess=args.preprocess, args_only_test=args.Test,
                            args_epochs=args.epochs, args_patience=args.patience,
                            args_grid_search=args.grid_search_trials,
                            args_show_model=args.show_model)
    return config

