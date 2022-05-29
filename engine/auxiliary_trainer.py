# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging

import torch
import torch.nn as nn
import torch.optim as optim
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from utils.reid_metric import R1_mAP
from layers.auxiliary_loss_weight import AuxiliaryWeight, param_grad_dot

global ITER
ITER = 0

def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img)
        loss = loss_fn(score, feat, target)
        loss.backward()
        optimizer.step()
        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_trainer_with_center(cetner_loss_weight, model_structure, model_structure_twin, auxiliary_weight,
                                          target_start=0, device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    model, loss_fn, center_criterion, optimizer, optimizer_center, scheduler = model_structure
    model_twin, loss_fn_twin, center_criterion_twin, optimizer_twin, optimizer_center_twin, scheduler_twin = model_structure_twin
    if True:
        pass
        # loss_fn_twin, center_criterion_twin = loss_fn, center_criterion
    aux_optim = optim.AdamW(auxiliary_weight.parameters(), lr=0.001, betas=[0.5, 0.9], weight_decay=0.2)

    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            model_twin = nn.DataParallel(model_twin)
            auxiliary_weight = nn.DataParallel(auxiliary_weight)
        model.to(device)
        model_twin.to(device)
        auxiliary_weight.to(device)
    grad_list = []

    def _update(engine, batch):
        # prepare data
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target

        # copy model
        model_twin.load_state_dict(model.state_dict())
        center_criterion_twin.load_state_dict(center_criterion.state_dict())

        model.train()
        model_twin.train()
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        optimizer_twin.zero_grad()
        optimizer_center_twin.zero_grad()

        # update twin model with full img and loss
        model.zero_grad()
        model_twin.zero_grad()
        with torch.no_grad():
            w = auxiliary_weight()

        if w.shape[0] == 1:
            source_mask = target < target_start
            target_mask = target >= target_start
            mask = source_mask * w + target_mask
        else:
            source_mask = target < target_start
            target_mask = target >= target_start
            mask = w[target * source_mask] * source_mask + target_mask

        score_twin, feat_twin = model_twin(img)
        loss_twin = loss_fn_twin(score_twin, feat_twin, target)
        # print("Total loss is {}, center loss is {}".format(loss, center_criterion(feat, target)))
        loss_twin = (loss_twin * mask).mean()
        loss_twin.backward()
        optimizer_twin.step()

        # update auxiliary weight
        w = auxiliary_weight()
        score, feat = model(img)
        score_twin, feat_twin = model_twin(img)
        loss_main = (loss_fn(score, feat, target) * target_mask).mean()
        loss_main_twin = (loss_fn(score_twin, feat_twin, target) * target_mask).mean()
        model.zero_grad()
        model_twin.zero_grad()
        loss_main.backward()
        loss_main_twin.backward()

        update_w = True
        if w.shape[0] == 1:
            loss_w = w * param_grad_dot(model.classifier.weight.grad, model_twin.classifier.weight.grad)
        else:
            loss_w = sum([w[t] for t, m in zip(target, source_mask) if m]) * 20 * \
                     param_grad_dot(model.classifier.weight.grad, model_twin.classifier.weight.grad) + \
                     sum(w) * .1
            grad = param_grad_dot(model.classifier.weight.grad, model_twin.classifier.weight.grad).item()
            grad_list.append(grad)
            if grad < 0:
                pass
            if int(sum(source_mask)) == 0:
                update_w = False

        loss_w *= -1

        if update_w:
            aux_optim.zero_grad()
            loss_w.backward()
            aux_optim.step()

        # update model
        model.zero_grad()
        model_twin.zero_grad()
        optimizer_center.zero_grad()
        with torch.no_grad():
            w = auxiliary_weight()
        if w.shape[0] == 1:
            source_mask = target < target_start
            target_mask = target >= target_start
            mask = source_mask * w + target_mask
        else:
            source_mask = target < target_start
            target_mask = target >= target_start
            mask = w[target * source_mask] * source_mask + target_mask
        score, feat = model(img)
        loss = loss_fn(score, feat, target)
        loss = (loss * mask).mean()
        # print("Total loss is {}, center loss is {}".format(loss, center_criterion(feat, target)))
        loss.backward()
        optimizer.step()
        optimizer_center.step()

        # for param in center_criterion.parameters():
        #     param.grad.data *= (1. / cetner_loss_weight)
        # optimizer_center.step()

        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)


def do_train_with_center(
        cfg,
        train_loader,
        val_loader,
        num_query,
        start_epoch,
        target_start,
        model_structure,
        model_structure_twin
):
    model, loss_func, center_criterion, optimizer, optimizer_center, scheduler = model_structure
    model_twin, loss_func_twin, center_criterion_twin, optimizer_twin, optimizer_center_twin, scheduler_twin = model_structure_twin

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")

    # auxiliary_weight = AuxiliaryWeight(size=target_start)
    auxiliary_weight = AuxiliaryWeight()

    trainer = create_supervised_trainer_with_center(cfg.SOLVER.CENTER_LOSS_WEIGHT,
                                                    model_structure, model_structure_twin,
                                                    auxiliary_weight,
                                                    target_start, device=device)

    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer,
                                                                     'center_param': center_criterion,
                                                                     'optimizer_center': optimizer_center})

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()
        scheduler_twin.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}, W: {:.3f}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0], auxiliary_weight.mean().item()))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)
