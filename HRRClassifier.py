import torch
import torch.nn as nn
import HRRTorch


def MLP_classifier(feature_size, layer_sizes, nonlinearity=nn.LeakyReLU()):
    model = nn.Sequential()
    numneurons = feature_size
    for i, size in enumerate(layer_sizes):
        model.add_module(str(i) + 'Linear', nn.Linear(numneurons, size))
        model.add_module(str(i) + 'Nonlinear', nonlinearity)
        numneurons = size
    model.add_module(str(len(layer_sizes)) + 'Linear',
                     nn.Linear(numneurons, 1))
    return model


class Cat_featurizer(nn.Module):
    def __init__(self, hrr_size):
        super(Cat_featurizer, self).__init__()

        self.output_size = hrr_size * 2

    def forward(self, problemhrr, lemmahrr):
        return torch.cat([problemhrr, lemmahrr])

    def get_output_size(self):
        return self.output_size


class Decoder_featurizer(nn.Module):
    def __init__(self, hrr_size, num_decoders):
        super(Decoder_featurizer, self).__init__()

        decoders = [nn.Parameter(torch.Tensor(hrr_size))
                    for i in range(num_decoders)]
        self.decoders = nn.ParameterList(decoders)

        self.output_size = hrr_size * 2 * (1 + num_decoders)

        for p in self.decoders:
            nn.init.normal_(p)
            with torch.no_grad():
                p /= p.norm()

    def forward(self, problemhrr, lemmahrr):
        return torch.cat([
            problemhrr,
            lemmahrr,
            *(HRRTorch.associate(decoder, problemhrr)
              for decoder in self.decoders),
            *(HRRTorch.associate(decoder, lemmahrr)
              for decoder in self.decoders),
        ])

    def get_output_size(self):
        return self.output_size


class DecoderComp_featurizer(nn.Module):
    def __init__(self, hrr_size, num_decoders):
        super(DecoderComp_featurizer, self).__init__()

        decoders = [nn.Parameter(torch.empty((2, hrr_size)))
                    for i in range(num_decoders)]
        self.decoders = nn.ParameterList(decoders)

        self.output_size = 2 * hrr_size * 2 * (1 + num_decoders)

        for p in self.decoders:
            nn.init.normal_(p)
            with torch.no_grad():
                p /= p.norm()

    def forward(self, problemhrr, lemmahrr):
        return torch.cat([
            problemhrr,
            lemmahrr,
            *(HRRTorch.associate_comp(decoder, problemhrr)
              for decoder in self.decoders),
            *(HRRTorch.associate_comp(decoder, lemmahrr)
              for decoder in self.decoders),
        ]).reshape(-1)

    def get_output_size(self):
        return self.output_size


class HRRClassifier(nn.Module):

    def __init__(self, hrr_size, hrrmodel, featurizer, classifier):
        super(HRRClassifier, self).__init__()

        self.hrr_size = hrr_size

        self.hrrmodel = hrrmodel
        self.featurizer = featurizer
        self.classifier = classifier

    def forward(self, points):
        out = []
        for problem, lemma in points:
            problemhrr = self.hrrmodel(
                torch.zeros(self.hrr_size).detach(), problem)
            lemmahrr = self.hrrmodel(
                torch.zeros(self.hrr_size).detach(), lemma)
            features = self.featurizer(problemhrr, lemmahrr)
            out.append(self.classifier(features))
        return torch.cat(out)


class HRRClassifierLoss(nn.Module):
    def __init__(self, model, experimentsettings):
        super(HRRClassifierLoss, self).__init__()

        self.model = model
        self.experimentsettings = experimentsettings
        self.bceloss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        errorloss = self.bceloss(pred, target)

        # Add regularisation for unit vectors (should have norm 1)
        unitvecloss = torch.tensor(0.)
        count = 0
        for param in (
                *self.model.hrrmodel.fixed_encodings.values(),
                # *self.model.featurizer.decoders,
        ):
            sqsum = torch.sum(param ** 2)
            # Sum squared error = (norm - 1) ** 2
            #                   = (norm**2 - 2*norm + 1)
            unitvecloss += sqsum - 2 * sqsum ** 0.5 + 1
            count += 1
        unitvecloss /= count

        # Add regularisation for mixing vectors (should sum to 1)
        sumvecloss = torch.tensor(0.)
        count = 0
        for param in (
                *self.model.hrrmodel.ground_vec_merge_ratios.values(),
                self.model.hrrmodel.func_weights,
                self.model.hrrmodel.eq_weights,
                self.model.hrrmodel.disj_weights,
                self.model.hrrmodel.conj_weights,
                *self.model.hrrmodel.varmodel.parameters(),
                *self.model.hrrmodel.constmodel.parameters(),
                *self.model.hrrmodel.distmodel.parameters(),
                *self.model.hrrmodel.funcmodel.parameters(),
                *self.model.hrrmodel.eqmodel.parameters(),
                *self.model.hrrmodel.disjmodel.parameters(),
                *self.model.hrrmodel.conjmodel.parameters(),
        ):
            sum_ = torch.sum(param)
            # Sum squared error = (sum - 1) ** 2
            sumvecloss += (sum_ - 1) ** 2
            count += 1
        sumvecloss /= count

        # Add regularisation for MLP weights
        recnetweightloss = torch.tensor(0.)
        count = 0
        for param in (
                *self.model.hrrmodel.varmodel.parameters(),
                *self.model.hrrmodel.constmodel.parameters(),
                *self.model.hrrmodel.distmodel.parameters(),
                *self.model.hrrmodel.funcmodel.parameters(),
                *self.model.hrrmodel.eqmodel.parameters(),
                *self.model.hrrmodel.disjmodel.parameters(),
                *self.model.hrrmodel.conjmodel.parameters(),
        ):
            recnetweightloss += param.norm()
            count += 1
        recnetweightloss /= count

        # Add regularisation for variance vectors?
        # Let's not bother (looking at the data, it looks well behaved)

        # Add regularisation for MLP weights
        mlpweightloss = torch.tensor(0.)
        count = 0
        for param in self.model.classifier.parameters():
            mlpweightloss += param.norm()
            count += 1
        mlpweightloss /= count

        allweightloss = torch.tensor(0.)
        for param in self.model.parameters():
            allweightloss += param.sum()

        return (errorloss + 0 * allweightloss,
                self.experimentsettings.unitvecloss_weight * unitvecloss +
                self.experimentsettings.sumvecloss_weight * sumvecloss +
                self.experimentsettings.mlpweightloss_weight * mlpweightloss +
                self.experimentsettings.recnetweightloss_weight * recnetweightloss
                )
