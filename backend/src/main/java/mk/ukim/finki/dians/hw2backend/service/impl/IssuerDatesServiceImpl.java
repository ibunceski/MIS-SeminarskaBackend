package mk.ukim.finki.dians.hw2backend.service.impl;

import mk.ukim.finki.dians.hw2backend.model.IssuerDates;
import mk.ukim.finki.dians.hw2backend.repository.IssuerDatesRepository;
import mk.ukim.finki.dians.hw2backend.service.IssuerDatesService;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Service
public class IssuerDatesServiceImpl implements IssuerDatesService {

    final IssuerDatesRepository issuerDatesRepository;

    public IssuerDatesServiceImpl(IssuerDatesRepository issuerDatesRepository) {
        this.issuerDatesRepository = issuerDatesRepository;
    }

    @Override
    public Optional<IssuerDates> getLastDateForIssuer(String issuer) {
        return issuerDatesRepository.findByIssuer(issuer);
    }
}
